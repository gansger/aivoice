"""Entry point: python -m aivoice"""
import ctypes
import faulthandler
import logging
import sys
from ctypes import wintypes

from . import APP_DIR, APP_NAME
from .ui import MainWindow, build_app

ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
ERROR_ALREADY_EXISTS = 183
SW_RESTORE = 9
LOG_PATH = APP_DIR / "aivoice.log"


def _claim_single_instance():
    """True if no other copy runs. Two would fight over the same output and confuse OBS's window match.

    A named mutex rather than a lock file: Windows releases it even if the program crashes.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    # The handle stays open on purpose: holding it for the process lifetime is what claims the name.
    kernel32.CreateMutexW(None, False, f"Local\\{APP_NAME}")
    return ctypes.get_last_error() != ERROR_ALREADY_EXISTS


def _show_running_copy():
    user32 = ctypes.windll.user32
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]

    def check(hwnd, _):
        title, class_name = ctypes.create_unicode_buffer(64), ctypes.create_unicode_buffer(64)
        user32.GetWindowTextW(hwnd, title, 64)
        user32.GetClassNameW(hwnd, class_name, 64)
        # The class check skips an Explorer window showing a folder called AIVoice.
        if title.value == APP_NAME and class_name.value.startswith("Qt"):
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
            return False
        return True

    user32.EnumWindows(ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(check), 0)


def main():
    if not _claim_single_instance():
        _show_running_copy()
        return 0

    log_file = open(LOG_PATH, "w", encoding="utf-8", buffering=1)
    faulthandler.enable(log_file)
    logging.basicConfig(stream=log_file, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.excepthook = lambda kind, value, trace: logging.error("Unhandled exception", exc_info=(kind, value, trace))

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS)
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_NAME)

    app = build_app(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
