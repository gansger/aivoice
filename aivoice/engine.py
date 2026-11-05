"""Real-time audio: microphone -> processing thread -> output device(s).

Callbacks only move samples through FIFOs. All the work happens on one high-priority thread,
so a slow model frame can't stall the audio devices.
"""
import ctypes
import gc
import threading
import time

import numpy as np
import sounddevice as sd

from . import FRAME_SIZE, SAMPLE_RATE
from .denoise import ENGINES
from .dsp import Pipeline

# Steam's virtual mic copies raw bytes between mismatched formats: pitch drops by half and level jumps 30 dB.
BROKEN_OUTPUTS = ("Steam Streaming Microphone",)
SILENT_OUTPUTS = ("CABLE Input", "Steam Streaming Speakers")
VIRTUAL_INPUTS = ("Steam Streaming Microphone", "CABLE Output")

INPUT_STALL_SECONDS = 2.0
MAX_INPUT_BACKLOG = FRAME_SIZE * 10
DRIFT_TOLERANCE = FRAME_SIZE // 2
DRIFT_STEP = 4


def _wasapi_index():
    return next(i for i, api in enumerate(sd.query_hostapis()) if api["name"] == "Windows WASAPI")


def refresh_devices():
    """Re-enumerate so plugged and unplugged devices show up. Only call with no streams open."""
    sd._terminate()
    sd._initialize()


def list_devices(kind):
    """Names of the WASAPI devices for kind 'input' or 'output'."""
    wasapi = _wasapi_index()
    key = f"max_{kind}_channels"
    names = [d["name"] for d in sd.query_devices() if d["hostapi"] == wasapi and d[key] > 0]
    if kind == "output":
        names = [n for n in names if not any(bad in n for bad in BROKEN_OUTPUTS)]
    return names


def default_input():
    wasapi = sd.query_hostapis(_wasapi_index())
    names = list_devices("input")
    index = wasapi["default_input_device"]
    if index >= 0:
        name = sd.query_devices(index)["name"]
        if not any(v in name for v in VIRTUAL_INPUTS):
            return name
    real = [n for n in names if not any(v in n for v in VIRTUAL_INPUTS)]
    return real[0] if real else ""


def default_output():
    """Prefer a device nobody listens to, so the voice reaches OBS but not the speakers."""
    names = list_devices("output")
    for wanted in SILENT_OUTPUTS:
        for name in names:
            if wanted in name:
                return name
    return ""


def is_cable(name):
    return "CABLE Input" in name


def _device_index(name, kind):
    wasapi = _wasapi_index()
    key = f"max_{kind}_channels"
    for index, device in enumerate(sd.query_devices()):
        if device["hostapi"] == wasapi and device[key] > 0 and device["name"] == name:
            return index, device[key]
    raise LookupError(f"Устройство не найдено: {name}")


class Fifo:
    """Mono sample FIFO shared between a stream callback and the processing thread."""

    def __init__(self, capacity):
        self._data = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._read = 0
        self.available = 0
        self._lock = threading.Lock()

    def write(self, x):
        with self._lock:
            n = x.size
            if n >= self._capacity:
                x, n = x[-self._capacity:], self._capacity
            start = (self._read + self.available) % self._capacity
            first = min(n, self._capacity - start)
            self._data[start:start + first] = x[:first]
            self._data[:n - first] = x[first:]
            overflow = self.available + n - self._capacity
            if overflow > 0:
                self._read = (self._read + overflow) % self._capacity
            self.available = min(self._capacity, self.available + n)

    def read(self, n):
        with self._lock:
            n = min(n, self.available)
            first = min(n, self._capacity - self._read)
            output = np.concatenate((self._data[self._read:self._read + first], self._data[:n - first]))
            self._read = (self._read + n) % self._capacity
            self.available -= n
            return output


class OutputSink:
    """One output stream with its FIFO. Corrects clock drift against the mic in tiny resampling steps."""

    FADE_SAMPLES = 240

    def __init__(self, target_samples, on_error):
        self.target = target_samples
        self.fifo = Fifo(SAMPLE_RATE)
        self.underruns = 0
        self.device_glitches = 0
        self.corrections = 0
        self._priming = True
        self._fade_in = False
        self._smoothed = float(target_samples)
        self._on_error = on_error
        self.stream = None

    def open(self, name):
        index, channels = _device_index(name, "output")
        channels = min(channels, 2)
        self.stream = sd.OutputStream(device=index, samplerate=SAMPLE_RATE, channels=channels, dtype="float32", blocksize=FRAME_SIZE,
                                      latency="low", callback=self._callback, finished_callback=self._on_error,
                                      extra_settings=sd.WasapiSettings(auto_convert=True))
        self.stream.start()

    def _callback(self, outdata, frames, time_info, status):
        if status:
            self.device_glitches += 1
        if self._priming:
            if self.fifo.available < self.target:
                outdata.fill(0)
                return
            self._priming = False
            self._fade_in = True
        x = self.fifo.read(frames)
        if x.size < frames:
            self.underruns += 1
            self._priming = True
            outdata.fill(0)
            if x.size:
                x = x * np.linspace(1.0, 0.0, x.size, dtype=np.float32)
                outdata[:x.size, :] = x[:, None]
            return
        if self._fade_in:
            self._fade_in = False
            x = x.copy()
            x[:self.FADE_SAMPLES] *= np.linspace(0.0, 1.0, self.FADE_SAMPLES, dtype=np.float32)
        outdata[:] = x[:, None]

    def push(self, frame):
        self._smoothed += (self.fifo.available - self._smoothed) * 0.01
        error = self._smoothed - self.target
        if abs(error) > DRIFT_TOLERANCE and not self._priming:
            size = frame.size - DRIFT_STEP if error > 0 else frame.size + DRIFT_STEP
            self.corrections += 1
            frame = np.interp(np.linspace(0, frame.size - 1, size), np.arange(frame.size), frame).astype(np.float32)
        self.fifo.write(frame)

    def close(self):
        if self.stream is not None:
            self.stream.close(ignore_errors=True)
            self.stream = None


class Engine:
    """Owns the streams and the processing thread. Controlled from the UI thread."""

    def __init__(self, settings):
        self.settings = settings
        self.pipeline = None
        self.error = ""
        self.load = 0.0
        self.dropped = 0
        self.latency_ms = 0.0
        self._input_fifo = None
        self._input_stream = None
        self._sink = None
        self._monitor = None
        self._monitor_lock = threading.Lock()
        self._data_ready = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._last_input = 0.0
        self._failed = False

    @property
    def running(self):
        return self._thread is not None

    @property
    def underruns(self):
        return self._sink.underruns + self._sink.device_glitches if self._sink else 0

    @property
    def needs_restart(self):
        """True when a device vanished or stopped delivering while we are meant to be running."""
        if not self.running:
            return False
        return self._failed or time.monotonic() - self._last_input > INPUT_STALL_SECONDS

    def start(self):
        self.stop()
        self.error = ""
        self._failed = False
        self.dropped = 0
        settings = self.settings
        try:
            if not settings.input_device or not settings.output_device:
                raise LookupError("Выберите микрофон и выход")
            denoiser = ENGINES.get(settings.engine, ENGINES["dpdfnet"])()
            self.pipeline = Pipeline(denoiser, settings)
            self._input_fifo = Fifo(SAMPLE_RATE)
            self._sink = OutputSink(int(SAMPLE_RATE * settings.buffer_ms / 1000), self._on_stream_finished)
            self._stop.clear()
            self._data_ready.clear()
            self._last_input = time.monotonic()
            self._thread = threading.Thread(target=self._run, name="aivoice-audio", daemon=True)
            self._thread.start()
            self._sink.open(settings.output_device)
            self._open_input()
            output_latency = self._sink.stream.latency
            self.latency_ms = (self._input_latency() + output_latency + (denoiser.delay + FRAME_SIZE) / SAMPLE_RATE + settings.buffer_ms / 1000) * 1000
            gc.collect()
            gc.freeze()  # keeps long garbage-collector pauses off the audio thread
            return True
        except Exception as exc:
            self.stop()
            self.error = str(exc)
            return False

    def _open_input(self):
        index, channels = _device_index(self.settings.input_device, "input")
        self._input_stream = sd.InputStream(device=index, samplerate=SAMPLE_RATE, channels=min(channels, 2), dtype="float32", blocksize=FRAME_SIZE,
                                            latency="low", callback=self._input_callback, finished_callback=self._on_stream_finished,
                                            extra_settings=sd.WasapiSettings(auto_convert=True))
        self._input_stream.start()

    def _input_latency(self):
        return self._input_stream.latency

    def stop(self):
        self._stop.set()
        self._data_ready.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2)
        if self._input_stream is not None:
            self._input_stream.close(ignore_errors=True)
            self._input_stream = None
        self.set_monitor("")
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    def set_monitor(self, name):
        """Start or stop listening to the processed voice on another output. Empty name stops it."""
        with self._monitor_lock:
            old, self._monitor = self._monitor, None
        if old is not None:
            old.close()
        if not name or not self.running:
            return
        sink = OutputSink(int(SAMPLE_RATE * self.settings.buffer_ms / 1000), None)
        sink.open(name)
        with self._monitor_lock:
            self._monitor = sink

    def _on_stream_finished(self):
        if not self._stop.is_set():
            self._failed = True

    def _input_callback(self, indata, frames, time_info, status):
        self._last_input = time.monotonic()
        if status:
            self.dropped += 1
        mono = indata[:, 0].copy() if indata.shape[1] == 1 else indata.mean(axis=1)
        self._input_fifo.write(mono)
        self._data_ready.set()

    def _run(self):
        _raise_thread_priority()
        fifo, pipeline, sink = self._input_fifo, self.pipeline, self._sink
        budget = FRAME_SIZE / SAMPLE_RATE
        try:
            while not self._stop.is_set():
                if fifo.available < FRAME_SIZE:
                    self._data_ready.wait(0.1)
                    self._data_ready.clear()
                    continue
                if fifo.available > MAX_INPUT_BACKLOG:
                    # We fell behind (CPU starved): drop the backlog rather than keep the added delay forever.
                    skipped = fifo.available - FRAME_SIZE
                    fifo.read(skipped)
                    self.dropped += skipped // FRAME_SIZE
                started = time.perf_counter()
                output = pipeline.process(fifo.read(FRAME_SIZE))
                self.load += ((time.perf_counter() - started) / budget - self.load) * 0.02
                sink.push(output)
                with self._monitor_lock:
                    monitor = self._monitor
                if monitor is not None:
                    monitor.push(output)
        except Exception as exc:
            self.error = str(exc)
            self._failed = True


def _raise_thread_priority():
    """Ask Windows to treat this thread as pro audio, so a game under load can't starve it."""
    try:
        task = ctypes.c_ulong(0)
        avrt = ctypes.windll.avrt
        avrt.AvSetMmThreadCharacteristicsW.restype = ctypes.c_void_p
        avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(task))
    except OSError:
        pass
