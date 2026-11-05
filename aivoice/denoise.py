"""Neural noise suppressors. Each takes and returns one 480-sample float32 frame at 48 kHz."""
import ctypes
from pathlib import Path

import numpy as np

from . import FRAME_SIZE

ASSETS = Path(__file__).resolve().parent / "assets"


def _vorbis_window(size):
    s = np.sin(0.5 * np.pi * (np.arange(size) + 0.5) / (size / 2))
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


class DpdfNet:
    """DPDFNet-2 48 kHz (Ceva, Apache-2.0): causal STFT around a stateful ONNX model."""

    label = "DPDFNet — лучшее качество"
    delay = 2400  # measured samples between input and output (tools/evaluate.py); keeps the dry signal aligned when blending

    def __init__(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(str(ASSETS / "dpdfnet2_48khz_hr.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        self._in_spec, self._in_state = inputs[0].name, inputs[1].name
        self._outputs = [outputs[0].name, outputs[1].name]

        meta = self._session.get_modelmeta().custom_metadata_map
        erb = np.array([float(x) for x in meta["erb_norm_init"].split(",")], dtype=np.float32)
        spec = np.array([float(x) for x in meta["spec_norm_init"].split(",")], dtype=np.float32)
        self._state = np.zeros(int(meta["state_size"]), dtype=np.float32)
        erb_size = int(meta["erb_norm_state_size"])
        self._state[:erb_size] = erb
        self._state[erb_size:erb_size + int(meta["spec_norm_state_size"])] = spec

        self._window = _vorbis_window(FRAME_SIZE * 2)
        self._previous = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._overlap = np.zeros(FRAME_SIZE, dtype=np.float32)

    def process(self, frame):
        spectrum = np.fft.rfft(np.concatenate((self._previous, frame)) * self._window)
        self._previous = frame.copy()
        model_input = np.stack((spectrum.real, spectrum.imag), axis=-1).astype(np.float32)[None, None]
        enhanced, self._state = self._session.run(self._outputs, {self._in_spec: model_input, self._in_state: self._state})
        ri = enhanced[0, 0]
        time_frame = (np.fft.irfft(ri[:, 0] + 1j * ri[:, 1], n=FRAME_SIZE * 2) * self._window).astype(np.float32)
        output = self._overlap + time_frame[:FRAME_SIZE]
        self._overlap = time_frame[FRAME_SIZE:]
        return output


class RnNoise:
    """RNNoise (Xiph, BSD): much lighter on the CPU, for when a game leaves no room for DPDFNet."""

    label = "RNNoise — минимальная нагрузка"
    delay = 960

    def __init__(self):
        self._lib = ctypes.CDLL(str(ASSETS / "rnnoise.dll"))
        self._lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        self._lib.rnnoise_create.restype = ctypes.c_void_p
        self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self._lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self._lib.rnnoise_process_frame.restype = ctypes.c_float
        self._state = self._lib.rnnoise_create(None)

    def process(self, frame):
        buffer = frame * 32768.0  # RNNoise works on the 16-bit sample range
        pointer = buffer.ctypes.data
        self._lib.rnnoise_process_frame(self._state, pointer, pointer)
        return buffer / 32768.0

    def __del__(self):
        state, self._state = getattr(self, "_state", None), None
        if state:
            self._lib.rnnoise_destroy(state)


ENGINES = {"dpdfnet": DpdfNet, "rnnoise": RnNoise}
