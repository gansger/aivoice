"""Block-based voice processing chain: gain, high-pass, denoise blend, gate, compressor, limiter."""
import math

import numpy as np

from . import FRAME_SIZE, SAMPLE_RATE

BLOCK_MS = 1000.0 * FRAME_SIZE / SAMPLE_RATE
LIMITER_CEILING = 10 ** (-1.0 / 20)
_SILENCE_DB = -120.0


def db_to_gain(db):
    return 10.0 ** (db / 20.0)


def peak_db(x):
    peak = float(np.abs(x).max())
    return 20.0 * math.log10(peak) if peak > 1e-6 else _SILENCE_DB


def rms_db(x):
    power = float(np.mean(x * x))
    return 10.0 * math.log10(power) if power > 1e-12 else _SILENCE_DB


def _smoothing(time_ms):
    return math.exp(-BLOCK_MS / time_ms)


def strength_to_dry_mix(strength):
    """Strength in percent -> share of the original signal left in. 50% is -20 dB, 100% removes it all."""
    if strength >= 100:
        return 0.0
    return db_to_gain(-40.0 * strength / 100.0)


class Ramp:
    """Moves a gain to its target across one block so changes never click."""

    def __init__(self, value=1.0):
        self.value = value

    def apply(self, x, target):
        if target == self.value:
            return x * target if target != 1.0 else x
        ramp = np.linspace(self.value, target, x.size, endpoint=False, dtype=np.float32)
        self.value = target
        return x * ramp


class HighPass:
    """2nd-order Butterworth high-pass, removes rumble and desk thumps."""

    def __init__(self):
        self._freq = None
        self._z1 = self._z2 = 0.0

    def process(self, x, freq):
        if freq <= 0:
            self._freq = None
            self._z1 = self._z2 = 0.0
            return x
        if freq != self._freq:
            self._freq = freq
            w = 2.0 * math.pi * freq / SAMPLE_RATE
            alpha = math.sin(w) / math.sqrt(2.0)
            a0 = 1.0 + alpha
            cos_w = math.cos(w)
            self._b0 = (1.0 + cos_w) / 2.0 / a0
            self._b1 = -(1.0 + cos_w) / a0
            self._a1 = -2.0 * cos_w / a0
            self._a2 = (1.0 - alpha) / a0
        b0, b1, a1, a2 = self._b0, self._b1, self._a1, self._a2
        z1, z2 = self._z1, self._z2
        output = []
        for sample in x.tolist():
            y = b0 * sample + z1
            z1 = b1 * sample - a1 * y + z2
            z2 = b0 * sample - a2 * y
            output.append(y)
        self._z1, self._z2 = z1, z2
        return np.array(output, dtype=np.float32)


class Gate:
    """Closes between phrases to catch what the denoiser leaves behind."""

    FLOOR = db_to_gain(-40.0)
    HOLD_BLOCKS = 25
    HYSTERESIS_DB = 4.0

    def __init__(self):
        self._ramp = Ramp()
        self._hold = 0
        self._gain = 1.0
        self._open = True
        self._attack = _smoothing(10.0)
        self._release = _smoothing(100.0)

    def process(self, x, threshold_db):
        if threshold_db <= -80.0:
            self._gain = 1.0
            self._open = True
            return self._ramp.apply(x, 1.0)
        level = rms_db(x)
        if level > (threshold_db - self.HYSTERESIS_DB if self._open else threshold_db):
            self._open = True
            self._hold = self.HOLD_BLOCKS
        elif self._hold > 0:
            self._hold -= 1
        else:
            self._open = False
        target, coefficient = (1.0, self._attack) if self._open else (self.FLOOR, self._release)
        self._gain = target + (self._gain - target) * coefficient
        return self._ramp.apply(x, self._gain)


class Compressor:
    """Evens out loudness. One amount (0..1) sets threshold, ratio and makeup together."""

    def __init__(self):
        self._ramp = Ramp()
        self._reduction = 0.0
        self._attack = _smoothing(10.0)
        self._release = _smoothing(150.0)

    def process(self, x, amount):
        if amount <= 0.0:
            self._reduction = 0.0
            return self._ramp.apply(x, 1.0)
        threshold = -14.0 - 14.0 * amount
        ratio = 1.0 + 3.0 * amount
        makeup = -threshold * (1.0 - 1.0 / ratio) * 0.5
        target = -max(0.0, rms_db(x) - threshold) * (1.0 - 1.0 / ratio)
        coefficient = self._attack if target < self._reduction else self._release
        self._reduction = target + (self._reduction - target) * coefficient
        return self._ramp.apply(x, db_to_gain(self._reduction + makeup))


class Limiter:
    """Last stage: keeps peaks under -1 dBFS so the stream never clips."""

    ATTACK_SAMPLES = 48

    def __init__(self):
        self._gain = 1.0

    def process(self, x):
        peak = float(np.abs(x).max())
        target = LIMITER_CEILING / peak if peak > LIMITER_CEILING else 1.0
        if target < self._gain:
            gains = np.full(x.size, target, dtype=np.float32)
            gains[:self.ATTACK_SAMPLES] = np.linspace(self._gain, target, self.ATTACK_SAMPLES, dtype=np.float32)
            self._gain = target
        else:
            released = target + (self._gain - target) * 0.9
            gains = np.linspace(self._gain, released, x.size, dtype=np.float32)
            self._gain = released
        return np.clip(x * gains, -LIMITER_CEILING, LIMITER_CEILING)


class Pipeline:
    """The whole chain for one microphone. `process` runs on the audio thread and reads `settings` live."""

    def __init__(self, denoiser, settings):
        self.denoiser = denoiser
        self.settings = settings
        self._input_ramp = Ramp(db_to_gain(settings.input_gain_db))
        self._output_ramp = Ramp(db_to_gain(settings.output_gain_db))
        self._mute_ramp = Ramp(0.0 if settings.muted else 1.0)
        self._highpass = HighPass()
        self._gate = Gate()
        self._compressor = Compressor()
        self._limiter = Limiter()
        self._dry_delay = np.zeros(denoiser.delay, dtype=np.float32)
        self._dry_mix = self._target_dry_mix()
        self.input_db = _SILENCE_DB
        self.output_db = _SILENCE_DB

    def _target_dry_mix(self):
        settings = self.settings
        return strength_to_dry_mix(settings.strength) if settings.enabled else 1.0

    def _denoise(self, x):
        # The model keeps running while bypassed, so switching on is instant and stays in time.
        enhanced = self.denoiser.process(x)
        delayed = np.concatenate((self._dry_delay, x))
        dry, self._dry_delay = delayed[:FRAME_SIZE], delayed[FRAME_SIZE:]
        target = self._target_dry_mix()
        mix = np.linspace(self._dry_mix, target, FRAME_SIZE, endpoint=False, dtype=np.float32)
        self._dry_mix = target
        return enhanced * (1.0 - mix) + dry * mix

    def process(self, frame):
        settings = self.settings
        x = self._input_ramp.apply(frame, db_to_gain(settings.input_gain_db))
        self.input_db = peak_db(x)
        x = self._highpass.process(x, settings.highpass_hz)
        x = self._denoise(x)
        x = self._gate.process(x, settings.gate_db)
        x = self._compressor.process(x, settings.compression / 100.0)
        x = self._output_ramp.apply(x, db_to_gain(settings.output_gain_db))
        x = self._mute_ramp.apply(x, 0.0 if settings.muted else 1.0)
        x = self._limiter.process(x)
        self.output_db = peak_db(x)
        return x
