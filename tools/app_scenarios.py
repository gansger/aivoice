"""Drives the real window through everyday situations using real devices and reports what happened.

Uses a temporary settings file. The mic is never saved or played aloud: monitoring goes to an unconnected S/PDIF output.
Usage: python tools/app_scenarios.py [soak_seconds]
"""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer

from aivoice import engine as audio
from aivoice import settings as store

store.SETTINGS_PATH = Path(tempfile.gettempdir()) / "aivoice_scenarios_settings.json"
store.SETTINGS_PATH.unlink(missing_ok=True)

from aivoice.ui import MainWindow, build_app

SOAK = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
app = build_app(sys.argv)
window = MainWindow()
window.show()
engine = window.engine
failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def later(ms):
    """Pump the event loop for a while, as if the user were just waiting."""
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def healthy():
    return engine.running and not engine.needs_restart and engine.pipeline.input_db > -200


later(1000)
check("starts with default devices", healthy(), f"{engine.settings.input_device} -> {engine.settings.output_device}")

# Soak: is drift correction bounded once settled, not constantly nudging?
later(5000)
first = engine._sink.corrections
start = time.monotonic()
later(SOAK * 1000)
rate = (engine._sink.corrections - first) / (time.monotonic() - start)
check("soak: no dropouts", engine.underruns + engine.dropped == 0, f"underruns {engine.underruns}, dropped {engine.dropped}, load {engine.load * 100:.0f}%")
check("soak: drift correction stays rare", rate < 1.0, f"{rate:.2f}/s over {SOAK:.0f} s, fifo {engine._sink.fifo.available} samples, target {engine._sink.target}")

# Engine switch, both ways, from the combo box a user would touch.
window.engine_combo.setCurrentIndex(window.engine_combo.findData("rnnoise"))
later(1500)
check("switch to RNNoise", healthy() and type(engine.pipeline.denoiser).__name__ == "RnNoise", f"load {engine.load * 100:.0f}%, latency {engine.latency_ms:.0f} ms")
window.engine_combo.setCurrentIndex(window.engine_combo.findData("dpdfnet"))
later(1500)
check("switch back to DPDFNet", healthy() and type(engine.pipeline.denoiser).__name__ == "DpdfNet")

# Monitoring to an output with nothing plugged in, so the mic isn't played aloud.
spdif = next((n for n in audio.list_devices("output") if "S/PDIF" in n), None)
if spdif:
    window.monitor_combo.setCurrentText(spdif)
    window.monitor_check.setChecked(True)
    later(1500)
    check("monitor on", engine._monitor is not None and engine._monitor.fifo.available > 0, f"fifo {engine._monitor.fifo.available if engine._monitor else None}")
    window.monitor_check.setChecked(False)
    later(300)
    check("monitor off", engine._monitor is None)
else:
    print("skip monitor test: no S/PDIF output")

# Device loss: the stream dying under us, as when USB is pulled.
old_sink = engine._sink
engine._input_stream.abort()
later(500)
check("device loss is noticed", engine.needs_restart, f"status: {window.status.text().strip()}")
later(4000)
check("recovers by itself", healthy() and engine._sink is not old_sink, f"status: {window.status.text().strip()}")

# Device that isn't there: must report it and keep trying, not crash.
good_input = engine.settings.input_device
engine.settings.input_device = "Нет такого микрофона"
window.restart_audio()
check("missing device is reported", not engine.running and "не найдено" in engine.error, engine.error)
later(500)
check("error shows in the window", "не найдено" in window.status.text(), window.status.text().strip())
engine.settings.input_device = good_input
later(4000)
check("recovers when device returns", healthy())

# Live controls don't disturb the stream.
before = engine.underruns + engine.dropped
window.power_button.click()
later(200)
check("bypass", engine.settings.enabled is False and window.power_button.text() == "Шумодав выключен")
window.power_button.click()
window.mute_button.click()
later(300)
check("mute silences output", engine.pipeline.output_db <= -119, f"{engine.pipeline.output_db:.0f} dB")
window.mute_button.click()
window._sliders["gate_db"][0].setValue(-50)
window._sliders["strength"][0].setValue(60)
later(800)
check("sliders reach the settings", (engine.settings.gate_db, engine.settings.strength) == (-50, 60))
check("settings saved", store.load(store.SETTINGS_PATH).strength == 60)
window._on_reset()
check("reset restores defaults", engine.settings.strength == 100 and engine.settings.gate_db == -80)
check("no dropouts from live changes", engine.underruns + engine.dropped == before, f"{engine.underruns + engine.dropped - before}")

window.close()
check("clean stop", not engine.running)
store.SETTINGS_PATH.unlink(missing_ok=True)
print("FAILED:" if failures else "ALL OK", failures or "")
sys.exit(1 if failures else 0)
