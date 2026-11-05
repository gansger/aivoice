import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aivoice import FRAME_SIZE, SAMPLE_RATE
from aivoice import settings as store
from aivoice.denoise import ENGINES
from aivoice.dsp import Compressor, Gate, HighPass, Limiter, Pipeline, rms_db, strength_to_dry_mix
from aivoice.engine import Fifo, OutputSink
from aivoice.settings import Settings


def tone(freq, amplitude, frames=100):
    t = np.arange(frames * FRAME_SIZE) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def blocks(x):
    return [x[i:i + FRAME_SIZE] for i in range(0, x.size, FRAME_SIZE)]


class PassThrough:
    delay = 0

    def process(self, frame):
        return frame.copy()


class HighPassTest(unittest.TestCase):
    def run_filter(self, freq):
        hp = HighPass()
        return np.concatenate([hp.process(b, 100) for b in blocks(tone(freq, 0.5))])[FRAME_SIZE * 20:]

    def test_cuts_rumble_keeps_voice(self):
        self.assertLess(rms_db(self.run_filter(20)), rms_db(tone(20, 0.5)) - 20)
        self.assertAlmostEqual(rms_db(self.run_filter(1000)), rms_db(tone(1000, 0.5)), delta=0.1)


class GateTest(unittest.TestCase):
    def test_closes_on_quiet_and_opens_on_loud(self):
        gate = Gate()
        quiet = [gate.process(b, -40) for b in blocks(tone(300, 0.001, 200))]
        self.assertLess(rms_db(quiet[-1]), rms_db(tone(300, 0.001)) - 30)
        loud = [gate.process(b, -40) for b in blocks(tone(300, 0.3, 30))]
        self.assertAlmostEqual(rms_db(loud[-1]), rms_db(tone(300, 0.3)), delta=0.5)

    def test_off_position_is_transparent(self):
        gate = Gate()
        x = tone(300, 0.001, 1)
        np.testing.assert_array_equal(gate.process(x, -80), x)


class CompressorTest(unittest.TestCase):
    def settled(self, amplitude, amount=1.0):
        compressor = Compressor()
        for b in blocks(tone(300, amplitude, 100)):
            output = compressor.process(b, amount)
        return rms_db(output)

    def test_narrows_dynamic_range(self):
        input_spread = rms_db(tone(300, 0.7)) - rms_db(tone(300, 0.07))
        self.assertLess(self.settled(0.7) - self.settled(0.07), input_spread * 0.7)


class LimiterTest(unittest.TestCase):
    def test_never_exceeds_ceiling_or_clicks_at_low_level(self):
        limiter = Limiter()
        output = np.concatenate([limiter.process(b) for b in blocks(tone(300, 3.0))])
        self.assertLessEqual(float(np.abs(output).max()), 10 ** (-1 / 20) + 1e-6)
        limiter = Limiter()
        x = tone(300, 0.2, 1)
        np.testing.assert_allclose(limiter.process(x), x, atol=1e-6)


class PipelineTest(unittest.TestCase):
    def settings(self, **overrides):
        return Settings(highpass_hz=0, gate_db=-80, compression=0, **overrides)

    def test_mute_silences_output(self):
        pipeline = Pipeline(PassThrough(), self.settings(muted=True))
        for b in blocks(tone(300, 0.3, 20)):
            output = pipeline.process(b)
        self.assertEqual(float(np.abs(output).max()), 0.0)

    def test_bypass_returns_dry_signal_in_time_with_delay(self):
        class Delayed(PassThrough):
            delay = 960

            def process(self, frame):
                return np.zeros_like(frame)  # a silent model proves we only hear the dry path

        x = tone(300, 0.3, 50)
        pipeline = Pipeline(Delayed(), self.settings(enabled=False))
        output = np.concatenate([pipeline.process(b) for b in blocks(x)])
        np.testing.assert_allclose(output[960:], x[:-960], atol=1e-6)

    def test_strength_curve(self):
        self.assertEqual(strength_to_dry_mix(100), 0.0)
        self.assertEqual(strength_to_dry_mix(0), 1.0)
        self.assertAlmostEqual(strength_to_dry_mix(50), 0.1, places=5)

    def test_real_engines_reduce_stationary_noise(self):
        noise = (np.random.default_rng(0).standard_normal(FRAME_SIZE * 400) * 0.02).astype(np.float32)
        for name, engine in ENGINES.items():
            with self.subTest(engine=name):
                pipeline = Pipeline(engine(), self.settings())
                output = [pipeline.process(b) for b in blocks(noise)]
                self.assertLess(rms_db(np.concatenate(output[200:])), rms_db(noise) - 15)


class FifoTest(unittest.TestCase):
    def test_wraps_and_keeps_order(self):
        fifo = Fifo(10)
        fifo.write(np.arange(7, dtype=np.float32))
        fifo.read(5)
        fifo.write(np.arange(7, 13, dtype=np.float32))
        np.testing.assert_array_equal(fifo.read(10), [5, 6, 7, 8, 9, 10, 11, 12])

    def test_overflow_drops_oldest(self):
        fifo = Fifo(4)
        fifo.write(np.arange(6, dtype=np.float32))
        np.testing.assert_array_equal(fifo.read(4), [2, 3, 4, 5])

    def test_short_read(self):
        fifo = Fifo(4)
        fifo.write(np.array([1.0], dtype=np.float32))
        self.assertEqual(fifo.read(3).size, 1)


class OutputSinkTest(unittest.TestCase):
    def pull(self, sink):
        out = np.zeros((FRAME_SIZE, 2), dtype=np.float32)
        sink._callback(out, FRAME_SIZE, None, None)
        return out

    def test_waits_for_buffer_then_counts_underrun(self):
        sink = OutputSink(FRAME_SIZE * 2, None)
        sink.push(np.ones(FRAME_SIZE, dtype=np.float32))
        self.assertEqual(float(self.pull(sink).max()), 0.0)  # still priming
        sink.push(np.ones(FRAME_SIZE, dtype=np.float32))
        self.assertEqual(float(self.pull(sink)[-1, 0]), 1.0)
        self.pull(sink)
        self.pull(sink)
        self.assertEqual(sink.underruns, 1)

    def test_drift_correction_holds_the_buffer_level(self):
        # Mic clock 0.1% faster than the output: without correction the buffer would keep growing.
        sink = OutputSink(FRAME_SIZE * 3, None)
        for _ in range(4):
            sink.push(np.ones(FRAME_SIZE, dtype=np.float32))
        levels = []
        for i in range(30000):
            sink.push(np.ones(FRAME_SIZE, dtype=np.float32))
            if i % 1000 != 0:  # one extra input block per 1000 blocks
                self.pull(sink)
            levels.append(sink.fifo.available)
        self.assertLess(max(levels[-5000:]), FRAME_SIZE * 3 + FRAME_SIZE * 2)
        self.assertEqual(sink.underruns, 0)


class SettingsTest(unittest.TestCase):
    def test_round_trip_and_rejects_bad_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s.json"
            store.save(Settings(strength=42, input_device="Мик"), path)
            loaded = store.load(path)
            self.assertEqual((loaded.strength, loaded.input_device), (42, "Мик"))
            path.write_text(json.dumps({"strength": 900, "gate_db": "x", "enabled": 1, "engine": 5, "nope": 1}), encoding="utf-8")
            loaded = store.load(path)
            self.assertEqual(loaded.strength, 100)
            self.assertEqual(loaded.gate_db, Settings().gate_db)
            self.assertIs(loaded.enabled, True)
            self.assertEqual(loaded.engine, "dpdfnet")
            path.write_text("{broken", encoding="utf-8")
            self.assertEqual(store.load(path), Settings())


if __name__ == "__main__":
    unittest.main()
