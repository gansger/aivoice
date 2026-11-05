"""Whole-chain test: the real window and real audio streams, with noisy speech swapped in for the microphone,
captured the way OBS Application Audio Capture does it. Reports quality, real latency and dropouts.

Usage: python tools/end_to_end.py speech_48k.wav [--stress] [--switch SECONDS] [--no-ui]
"""
import argparse
import multiprocessing
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import evaluate
from process_loopback import capture

from aivoice import FRAME_SIZE, SAMPLE_RATE
from aivoice import engine as audio
from aivoice.dsp import Pipeline

LEAD = 4 * SAMPLE_RATE


def burn(stop_at):
    while time.time() < stop_at:
        sum(i * i for i in range(10000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("speech")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--switch", type=float, default=None)
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()
    if args.switch:
        sys.setswitchinterval(args.switch)

    speech = evaluate.read_wav(args.speech)
    speech = speech / np.abs(speech).max() * 0.5
    clean = np.concatenate((np.zeros(LEAD, dtype=np.float32), speech))
    noise = evaluate.room_noise(clean.size, np.random.default_rng(1))
    noise *= np.sqrt(np.mean(speech ** 2) / np.mean(noise ** 2) / 10)
    noisy = clean + noise

    times, inject = [], {"pos": 0, "started": None}
    original_process = Pipeline.process

    def timed_process(self, frame):
        started = time.perf_counter()
        result = original_process(self, frame)
        times.append(time.perf_counter() - started)
        return result

    Pipeline.process = timed_process

    def fake_input(self, indata, frames, time_info, status):
        # Real device timing, known content.
        self._last_input = time.monotonic()
        if status:
            self.dropped += 1
        pos = inject["pos"]
        if pos == 0:
            inject["started"] = time.monotonic()
        chunk = noisy[pos:pos + frames]
        if chunk.size < frames:
            chunk = np.concatenate((chunk, np.zeros(frames - chunk.size, dtype=np.float32)))
        inject["pos"] = pos + frames
        self._input_fifo.write(chunk)
        self._data_ready.set()

    audio.Engine._input_callback = fake_input

    seconds = noisy.size / SAMPLE_RATE + 1.5
    result = {}
    engines = []

    def grab():
        data = capture(os.getpid(), seconds)
        result["end"] = time.monotonic()
        result["data"] = data
        # Read while still running: the window stops the engine on exit, which clears these.
        running = engines[0]
        result["stats"] = (running.underruns, running.dropped, running._sink.corrections, running.load)

    if args.stress:
        stop_at = time.time() + 20
        for _ in range(os.cpu_count()):
            multiprocessing.Process(target=burn, args=(stop_at,), daemon=True).start()
        print(f"stress: {os.cpu_count()} busy processes for 20 s", flush=True)

    thread = threading.Thread(target=grab)
    from aivoice import settings as store
    store.SETTINGS_PATH = Path(tempfile.gettempdir()) / "aivoice_e2e_settings.json"
    store.SETTINGS_PATH.unlink(missing_ok=True)

    if args.no_ui:
        settings = store.Settings(input_device=audio.default_input(), output_device=audio.default_output(), compression=0, highpass_hz=0)
        engine = audio.Engine(settings)
        assert engine.start(), engine.error
        engines.append(engine)
        thread.start()
        thread.join()
        window = None
    else:
        from PySide6.QtCore import QTimer
        from aivoice.ui import MainWindow, build_app
        app = build_app(sys.argv)
        window = MainWindow()
        window.settings.compression = 0
        window.settings.highpass_hz = 0
        window.show()
        engine = window.engine
        assert engine.running, engine.error
        engines.append(engine)
        thread.start()
        poll = QTimer(interval=100)
        poll.timeout.connect(lambda: thread.is_alive() or app.quit())
        poll.start()
        app.exec()
        thread.join()

    print(f"estimated latency {engine.latency_ms:.0f} ms; devices: {engine.settings.input_device} -> {engine.settings.output_device}")
    underruns, dropped, corrections, load = result["stats"]
    print(f"underruns+device glitches {underruns}, dropped {dropped}, drift corrections {corrections}, load {load * 100:.0f}%")
    block = np.array(times[50:]) * 1000
    print(f"block ms: mean {block.mean():.2f} p50 {np.percentile(block, 50):.2f} p99 {np.percentile(block, 99):.2f} max {block.max():.2f}; over budget {(block > 10).mean() * 100:.2f}%")

    captured = result["data"].mean(axis=1)
    if window:
        window.close()
    else:
        engine.stop()

    start = evaluate.find_delay(clean, captured, limit=captured.size - clean.size // 2)
    aligned = captured[start:start + clean.size]
    n = aligned.size
    capture_start = result["end"] - captured.size / SAMPLE_RATE
    latency = capture_start + start / SAMPLE_RATE - inject["started"]
    quiet = np.sqrt(np.mean(aligned[LEAD // 2:LEAD] ** 2))
    noise_rms = np.sqrt(np.mean(noisy[LEAD // 2:LEAD] ** 2))
    print(f"captured {captured.size / SAMPLE_RATE:.1f} s; processing+output latency {latency * 1000:.0f} ms (the mic stream's own latency comes on top)")
    print(f"noise reduced {20 * np.log10(noise_rms / max(quiet, 1e-9)):.1f} dB; SI-SDR {evaluate.si_sdr(clean[LEAD:n], aligned[LEAD:n]):.1f} dB (input {evaluate.si_sdr(clean[LEAD:n], noisy[LEAD:n]):.1f} dB)")

    # Whole-file SI-SDR collapses if the delay shifts at all, so also judge each second on its own alignment.
    # A clean shift leaves local quality intact; real sample loss doesn't.
    lags, local = [], []
    window = SAMPLE_RATE
    for begin in range(LEAD, clean.size - window, window):
        reference = clean[begin:begin + window]
        search = captured[start + begin - 4800:start + begin + window + 4800]
        if search.size < window + 9600:
            break
        lag = evaluate.find_delay(reference, search, limit=9600)
        lags.append(lag - 4800)
        local.append(evaluate.si_sdr(reference, search[lag:lag + window]))
    print("local lag by second (samples):", lags)
    print("local SI-SDR by second (dB):", [round(float(v), 1) for v in local])
    longest_zero, run = 0, 0
    for block in np.array_split(captured[start + LEAD:start + clean.size], (clean.size - LEAD) // 48):
        run = run + 1 if np.abs(block).max() < 1e-7 else 0
        longest_zero = max(longest_zero, run)
    print(f"longest digital silence inside speech: {longest_zero} ms")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
