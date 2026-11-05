"""AIVoice: real-time microphone noise suppression for streaming."""
import sys
from pathlib import Path

APP_NAME = "AIVoice"
# Where settings and the log live: next to AIVoice.exe when built, the project folder otherwise.
APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
SAMPLE_RATE = 48000
FRAME_SIZE = 480  # 10 ms, the native hop of both denoisers
