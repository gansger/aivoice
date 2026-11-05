"""Dev tool: capture one process's audio with the same Windows API OBS "Application Audio Capture" uses.

Lets tests check what OBS would receive without touching the user's OBS setup.
"""
import ctypes
import threading
import uuid
from ctypes import POINTER, WINFUNCTYPE, byref, c_int32, c_uint32, c_uint64, c_ulong, c_void_p, c_wchar_p, cast

import numpy as np

HRESULT = c_int32
RATE = 48000


def _guid(text):
    return (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(text).bytes_le)


IID_IUNKNOWN = bytes(_guid("00000000-0000-0000-C000-000000000046"))
IID_HANDLER = bytes(_guid("41D949AB-9862-444A-80F6-C261334DA5EB"))
IID_AGILE = bytes(_guid("94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90"))
IID_AUDIO_CLIENT = _guid("1CB9AD4C-DBFA-4c32-B178-C2F568A703B2")
IID_CAPTURE_CLIENT = _guid("C8ADBD64-E71E-48a0-A4DE-185C395CD317")


class _Blob(ctypes.Structure):
    _fields_ = [("cb", c_ulong), ("data", c_void_p)]


class _PropVariant(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort), ("r3", ctypes.c_ushort), ("blob", _Blob), ("pad", c_uint64)]


class _ActivationParams(ctypes.Structure):
    _fields_ = [("type", c_uint32), ("pid", c_uint32), ("mode", c_uint32)]


class _WaveFormat(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("tag", ctypes.c_ushort), ("channels", ctypes.c_ushort), ("rate", c_uint32), ("bytes_per_sec", c_uint32),
                ("align", ctypes.c_ushort), ("bits", ctypes.c_ushort), ("size", ctypes.c_ushort)]


def _method(obj, index, *argtypes):
    vtable = cast(cast(obj, POINTER(c_void_p))[0], POINTER(c_void_p))
    return WINFUNCTYPE(HRESULT, c_void_p, *argtypes)(vtable[index])


def _check(hr, what):
    if hr < 0:
        raise OSError(f"{what} failed: 0x{hr & 0xFFFFFFFF:08X}")


class _Handler:
    """Minimal IActivateAudioInterfaceCompletionHandler COM object."""

    def __init__(self):
        self.done = threading.Event()
        self._qi = WINFUNCTYPE(HRESULT, c_void_p, c_void_p, POINTER(c_void_p))(self._query)
        self._ref = WINFUNCTYPE(c_ulong, c_void_p)(lambda this: 1)
        self._completed = WINFUNCTYPE(HRESULT, c_void_p, c_void_p)(self._on_completed)
        self._vtable = (c_void_p * 4)(cast(self._qi, c_void_p), cast(self._ref, c_void_p), cast(self._ref, c_void_p), cast(self._completed, c_void_p))
        self._object = (c_void_p * 1)(cast(self._vtable, c_void_p))
        self.pointer = cast(self._object, c_void_p)

    def _query(self, this, iid, out):
        if ctypes.string_at(iid, 16) in (IID_IUNKNOWN, IID_HANDLER, IID_AGILE):
            out[0] = this
            return 0
        out[0] = None
        return ctypes.c_long(0x80004002).value

    def _on_completed(self, this, operation):
        self.done.set()
        return 0


def capture(pid, seconds):
    """Return (stereo float32 array, captured seconds of wall clock) of audio rendered by `pid`."""
    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
    params = _ActivationParams(1, pid, 0)  # PROCESS_LOOPBACK, include process tree
    prop = _PropVariant(vt=65, blob=_Blob(ctypes.sizeof(params), ctypes.addressof(params)))
    handler = _Handler()
    operation = c_void_p()
    activate = ctypes.windll.mmdevapi.ActivateAudioInterfaceAsync
    activate.argtypes = [c_wchar_p, c_void_p, c_void_p, c_void_p, POINTER(c_void_p)]
    activate.restype = HRESULT
    _check(activate("VAD\\Process_Loopback", byref(IID_AUDIO_CLIENT), byref(prop), handler.pointer, byref(operation)), "Activate")
    if not handler.done.wait(5):
        raise TimeoutError("audio interface activation timed out")

    status, client = HRESULT(), c_void_p()
    _check(_method(operation, 3, POINTER(HRESULT), POINTER(c_void_p))(operation, byref(status), byref(client)), "GetActivateResult")
    _check(status.value, "activation result")

    fmt = _WaveFormat(3, 2, RATE, RATE * 8, 8, 32, 0)  # IEEE float stereo
    flags = 0x00020000 | 0x00040000  # LOOPBACK | EVENTCALLBACK
    _check(_method(client, 3, c_int32, c_uint32, ctypes.c_int64, ctypes.c_int64, c_void_p, c_void_p)(client, 0, flags, 200000, 0, byref(fmt), None), "Initialize")
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateEventW.restype = c_void_p
    kernel32.WaitForSingleObject.argtypes = [c_void_p, c_uint32]
    event = kernel32.CreateEventW(None, 0, 0, None)
    _check(_method(client, 13, c_void_p)(client, event), "SetEventHandle")
    capture_client = c_void_p()
    _check(_method(client, 14, c_void_p, POINTER(c_void_p))(client, byref(IID_CAPTURE_CLIENT), byref(capture_client)), "GetService")
    _check(_method(client, 10)(client), "Start")

    get_buffer = _method(capture_client, 3, POINTER(c_void_p), POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint64), POINTER(c_uint64))
    release_buffer = _method(capture_client, 4, c_uint32)
    next_packet = _method(capture_client, 5, POINTER(c_uint32))

    chunks = []
    total = 0
    wanted = int(seconds * RATE)
    waits = 0
    while total < wanted and waits < seconds * 400:
        kernel32.WaitForSingleObject(event, 20)
        waits += 1
        size = c_uint32()
        while next_packet(capture_client, byref(size)) >= 0 and size.value:
            data, frames, buf_flags = c_void_p(), c_uint32(), c_uint32()
            _check(get_buffer(capture_client, byref(data), byref(frames), byref(buf_flags), None, None), "GetBuffer")
            if buf_flags.value & 0x2:  # AUDCLNT_BUFFERFLAGS_SILENT
                chunks.append(np.zeros((frames.value, 2), dtype=np.float32))
            else:
                chunks.append(np.frombuffer(ctypes.string_at(data, frames.value * 8), dtype=np.float32).reshape(-1, 2).copy())
            total += frames.value
            release_buffer(capture_client, frames.value)
    _method(client, 11)(client)  # Stop
    release = lambda obj: WINFUNCTYPE(c_ulong, c_void_p)(cast(cast(obj, POINTER(c_void_p))[0], POINTER(c_void_p))[2])(obj)
    release(capture_client)
    release(client)
    release(operation)
    kernel32.CloseHandle.argtypes = [c_void_p]
    kernel32.CloseHandle(event)
    return np.concatenate(chunks) if chunks else np.zeros((0, 2), dtype=np.float32)
