"""Measure denoiser quality on a speech recording mixed with synthetic room noise.

Usage: python tools/evaluate.py speech_48k.wav [out_dir]
Reports delay, noise reduction, SI-SDR and block timing. With out_dir, also writes the WAVs to listen to.
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aivoice import FRAME_SIZE, SAMPLE_RATE
from aivoice.denoise import ENGINES
from aivoice.dsp import Pipeline
from aivoice.settings import Settings

LEAD_SECONDS = 4
SNR_DB = 10


def read_wav(path):
    """Mono 16-bit 48 kHz WAV -> float32."""
    with wave.open(str(path), "rb") as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def room_noise(size, rng):
    """Fan (pink-ish), mains hum and keyboard clicks."""
    spectrum = np.fft.rfft(rng.standard_normal(size))
    spectrum /= np.maximum(np.fft.rfftfreq(size, 1 / SAMPLE_RATE), 40) ** 0.5
    fan = np.fft.irfft(spectrum, size)
    fan /= np.sqrt(np.mean(fan ** 2))
    t = np.arange(size) / SAMPLE_RATE
    hum = 0.3 * np.sin(2 * np.pi * 50 * t) + 0.15 * np.sin(2 * np.pi * 150 * t)
    clicks = np.zeros(size)
    for start in rng.integers(0, size - 480, int(size / SAMPLE_RATE * 4)):
        clicks[start:start + 480] += 4 * np.exp(-np.arange(480) / 40) * rng.standard_normal(480)
    return (fan + hum + clicks).astype(np.float32)


def si_sdr(reference, estimate):
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    scale = np.dot(estimate, reference) / np.dot(reference, reference)
    target = scale * reference
    return 10 * np.log10(np.sum(target ** 2) / np.sum((estimate - target) ** 2))


def find_delay(reference, signal, limit=9600):
    """Lag of `signal` behind `reference`, in samples, by FFT cross-correlation."""
    n = reference.size + signal.size
    corr = np.fft.irfft(np.fft.rfft(signal, n) * np.conj(np.fft.rfft(reference, n)), n)
    return int(np.argmax(corr[:limit]))


def run(engine, noisy, **overrides):
    settings = Settings(highpass_hz=0, compression=0, gate_db=-80)
    for key, value in overrides.items():
        setattr(settings, key, value)
    pipeline = Pipeline(ENGINES[engine](), settings)
    frames = noisy.size // FRAME_SIZE
    output = np.zeros(frames * FRAME_SIZE, dtype=np.float32)
    times = []
    for i in range(frames):
        frame = noisy[i * FRAME_SIZE:(i + 1) * FRAME_SIZE]
        started = time.perf_counter()
        output[i * FRAME_SIZE:(i + 1) * FRAME_SIZE] = pipeline.process(frame)
        times.append(time.perf_counter() - started)
    return output, np.array(times[50:]) * 1000


def main():
    speech = read_wav(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    speech = speech / np.abs(speech).max() * 0.5
    rng = np.random.default_rng(1)
    lead = LEAD_SECONDS * SAMPLE_RATE
    clean = np.concatenate((np.zeros(lead, dtype=np.float32), speech))
    noise = room_noise(clean.size, rng)
    speech_power = np.mean(speech ** 2)
    noise *= np.sqrt(speech_power / np.mean(noise ** 2) / 10 ** (SNR_DB / 10))
    noisy = clean + noise
    region = slice(lead, None)
    print(f"input: SNR {SNR_DB} dB, SI-SDR {si_sdr(clean[region], noisy[region]):.1f} dB, length {noisy.size / SAMPLE_RATE:.1f} s")
    if out_dir:
        out_dir.mkdir(exist_ok=True)
        write_wav(out_dir / "input_noisy.wav", noisy)

    noise_rms = np.sqrt(np.mean(noisy[lead // 2:lead] ** 2))
    for engine in ENGINES:
        for strength in (100, 50, 0):
            output, times = run(engine, noisy, strength=strength)
            delay = find_delay(clean[lead:], output[lead:]) if strength else 0
            aligned = output[delay:]
            size = aligned.size
            quiet = np.sqrt(np.mean(output[lead // 2:lead] ** 2))
            sdr = si_sdr(clean[lead:size], aligned[lead:])
            print(f"{engine:8s} strength {strength:3d}%: delay {delay:4d} samples ({delay / 48:.1f} ms), noise reduced {20 * np.log10(noise_rms / max(quiet, 1e-9)):5.1f} dB, "
                  f"SI-SDR {sdr:5.1f} dB, block mean {times.mean():.2f} ms p99 {np.percentile(times, 99):.2f} ms max {times.max():.2f} ms")
            if out_dir:
                write_wav(out_dir / f"{engine}_{strength}.wav", output)


if __name__ == "__main__":
    main()
