"""User settings, stored as JSON next to the program."""
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import APP_DIR

SETTINGS_PATH = APP_DIR / "settings.json"

# Allowed range of every numeric setting; a hand-edited file can't push audio out of bounds.
LIMITS = {
    "strength": (0, 100),
    "highpass_hz": (0, 200),
    "gate_db": (-80, -20),
    "compression": (0, 100),
    "input_gain_db": (-12, 24),
    "output_gain_db": (-12, 12),
    "buffer_ms": (20, 200),
}


@dataclass
class Settings:
    input_device: str = ""
    output_device: str = ""
    monitor_device: str = ""
    engine: str = "dpdfnet"
    enabled: bool = True
    muted: bool = False
    strength: int = 100
    highpass_hz: int = 80
    gate_db: int = -80  # -80 means off
    compression: int = 30
    input_gain_db: int = 0
    output_gain_db: int = 0
    buffer_ms: int = 30


def load(path=None):
    settings = Settings()
    try:
        data = json.loads(Path(path or SETTINGS_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if not isinstance(data, dict):
        return settings
    for field in fields(Settings):
        value = data.get(field.name)
        if value is None or isinstance(value, bool) != isinstance(getattr(settings, field.name), bool):
            continue
        if field.name in LIMITS:
            if not isinstance(value, (int, float)):
                continue
            low, high = LIMITS[field.name]
            value = int(min(high, max(low, value)))
        elif not isinstance(value, type(getattr(settings, field.name))):
            continue
        setattr(settings, field.name, value)
    return settings


def save(settings, path=None):
    path = Path(path or SETTINGS_PATH)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
