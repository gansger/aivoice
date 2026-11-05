"""Main window."""
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QSlider, QVBoxLayout, QWidget)

from . import APP_NAME, engine as audio
from . import settings as store
from .denoise import ENGINES

METER_FLOOR_DB = -60.0
METER_DECAY_DB = 1.2
RESTART_INTERVAL_MS = 3000
SAVE_DELAY_MS = 500

STYLE = """
* { font-family: 'Segoe UI'; font-size: 13px; color: #e8e8ee; }
QMainWindow, #root { background: #14151a; }
#card { background: #1d1f26; border-radius: 10px; }
#title { font-size: 22px; font-weight: 600; }
#heading { font-size: 12px; font-weight: 600; color: #9aa0b0; }
#hint { color: #9aa0b0; font-size: 12px; }
#error { color: #ff7b7b; font-size: 12px; }
#value { color: #c9cde0; }
QComboBox { background: #2a2d37; border: 1px solid #3a3e4b; border-radius: 6px; padding: 5px 8px; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background: #2a2d37; selection-background-color: #4a56c9; outline: none; }
QPushButton { background: #2a2d37; border: 1px solid #3a3e4b; border-radius: 6px; padding: 6px 12px; }
QPushButton:hover { background: #343844; }
QPushButton#power { font-size: 14px; font-weight: 600; padding: 10px 18px; border-radius: 8px; }
QPushButton#power:checked { background: #3d6df2; border-color: #3d6df2; color: white; }
QPushButton#mute:checked { background: #d9434e; border-color: #d9434e; color: white; }
QSlider::groove:horizontal { height: 4px; background: #3a3e4b; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #3d6df2; border-radius: 2px; }
QSlider::handle:horizontal { background: #e8e8ee; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; background: #2a2d37; border: 1px solid #3a3e4b; }
QCheckBox::indicator:checked { background: #3d6df2; border-color: #3d6df2; }
"""


class Meter(QWidget):
    """Horizontal level meter in dBFS with a falling display and an optional threshold mark."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._level = METER_FLOOR_DB
        self._mark = None

    def set_level(self, db):
        self._level = max(db, self._level - METER_DECAY_DB, METER_FLOOR_DB)
        self.update()

    def set_mark(self, db):
        self._mark = db
        self.update()

    def _position(self, db):
        return (min(db, 0.0) - METER_FLOOR_DB) / -METER_FLOOR_DB * self.width()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2a2d37"))
        painter.drawRoundedRect(self.rect(), 4, 4)
        filled = self._position(self._level)
        if filled > 0:
            gradient = QLinearGradient(0, 0, self.width(), 0)
            gradient.setColorAt(0.0, QColor("#2fbf71"))
            gradient.setColorAt(0.75, QColor("#2fbf71"))
            gradient.setColorAt(0.88, QColor("#e8c547"))
            gradient.setColorAt(1.0, QColor("#e5484d"))
            painter.setBrush(gradient)
            painter.drawRoundedRect(QRectF(0, 0, filled, self.height()), 4, 4)
        if self._mark is not None:
            painter.setBrush(QColor("#ffffff"))
            painter.drawRect(QRectF(self._position(self._mark) - 1, 0, 2, self.height()))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = store.load()
        self.engine = audio.Engine(self.settings)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(_make_icon())
        self._updating = False
        self._restart_wait = 0
        self._sliders = {}

        self._save_timer = QTimer(self, singleShot=True, interval=SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self._save)

        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addLayout(self._build_header())
        layout.addWidget(self._build_meters())
        columns = QHBoxLayout()
        columns.setSpacing(10)
        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(self._build_devices())
        left.addWidget(self._build_denoise())
        left.addStretch(1)
        columns.addLayout(left, 1)
        columns.addWidget(self._build_processing(), 1)
        layout.addLayout(columns)
        layout.addLayout(self._build_footer())

        self._fill_devices()
        self._sync_controls()
        self.setMinimumWidth(780)

        self._timer = QTimer(self, interval=33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.restart_audio()

    # ---- layout ----

    def _card(self, heading):
        card = QFrame(objectName="card")
        box = QVBoxLayout(card)
        box.setContentsMargins(14, 12, 14, 14)
        box.setSpacing(8)
        box.addWidget(QLabel(heading, objectName="heading"))
        return card, box

    def _build_header(self):
        row = QHBoxLayout()
        row.addWidget(QLabel(APP_NAME, objectName="title"))
        self.status = QLabel("", objectName="hint")
        row.addWidget(self.status, 1, Qt.AlignmentFlag.AlignVCenter)
        self.mute_button = QPushButton("Выключить микрофон", objectName="mute", checkable=True)
        self.mute_button.toggled.connect(self._on_mute)
        row.addWidget(self.mute_button)
        self.power_button = QPushButton("Шумодав включён", objectName="power", checkable=True)
        self.power_button.toggled.connect(self._on_power)
        row.addWidget(self.power_button)
        return row

    def _build_meters(self):
        card, box = self._card("УРОВНИ")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        self.input_meter, self.output_meter = Meter(), Meter()
        grid.addWidget(QLabel("Микрофон"), 0, 0)
        grid.addWidget(self.input_meter, 0, 1)
        grid.addWidget(QLabel("На выходе"), 1, 0)
        grid.addWidget(self.output_meter, 1, 1)
        grid.setColumnStretch(1, 1)
        box.addLayout(grid)
        return card

    def _build_devices(self):
        card, box = self._card("УСТРОЙСТВА")
        self.input_combo, self.output_combo = QComboBox(), QComboBox()
        for combo in (self.input_combo, self.output_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(20)
        self.input_combo.currentTextChanged.connect(lambda name: self._on_device("input_device", name))
        self.output_combo.currentTextChanged.connect(lambda name: self._on_device("output_device", name))
        box.addWidget(QLabel("Микрофон"))
        box.addWidget(self.input_combo)
        box.addWidget(QLabel("Выход"))
        box.addWidget(self.output_combo)
        self.obs_hint = QLabel("", objectName="hint", wordWrap=True)
        box.addWidget(self.obs_hint)
        refresh = QPushButton("Обновить список")
        refresh.clicked.connect(self._on_refresh)
        box.addWidget(refresh, 0, Qt.AlignmentFlag.AlignLeft)
        return card

    def _build_denoise(self):
        card, box = self._card("ШУМОПОДАВЛЕНИЕ")
        self.engine_combo = QComboBox()
        for key, cls in ENGINES.items():
            self.engine_combo.addItem(cls.label, key)
        self.engine_combo.currentIndexChanged.connect(self._on_engine)
        box.addWidget(self.engine_combo)
        box.addLayout(self._slider("strength", "Сила", 0, 100, lambda v: "максимум" if v >= 100 else f"−{v * 0.4:.0f} дБ шума"))
        return card

    def _build_processing(self):
        card, box = self._card("ОБРАБОТКА ГОЛОСА")
        box.addLayout(self._slider("input_gain_db", "Усиление микрофона", -12, 24, lambda v: f"{v:+d} дБ"))
        box.addLayout(self._slider("highpass_hz", "Срез низких", 0, 200, lambda v: "выкл" if v == 0 else f"{v} Гц", step=10))
        box.addLayout(self._slider("gate_db", "Шумовой гейт", -80, -20, lambda v: "выкл" if v <= -80 else f"{v} дБ"))
        box.addLayout(self._slider("compression", "Выравнивание громкости", 0, 100, lambda v: "выкл" if v == 0 else f"{v}%"))
        box.addLayout(self._slider("output_gain_db", "Громкость на выходе", -12, 12, lambda v: f"{v:+d} дБ"))
        box.addStretch(1)
        reset = QPushButton("Сбросить обработку")
        reset.clicked.connect(self._on_reset)
        box.addWidget(reset, 0, Qt.AlignmentFlag.AlignLeft)
        return card

    def _build_footer(self):
        row = QHBoxLayout()
        self.monitor_check = QCheckBox("Слышать себя в:")
        self.monitor_check.toggled.connect(self._apply_monitor)
        row.addWidget(self.monitor_check)
        self.monitor_combo = QComboBox()
        self.monitor_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.monitor_combo.setMinimumContentsLength(18)
        self.monitor_combo.currentTextChanged.connect(self._on_monitor_device)
        row.addWidget(self.monitor_combo)
        row.addStretch(1)
        row.addWidget(QLabel("Запас буфера"))
        self.buffer_combo = QComboBox()
        for ms in (20, 30, 50, 100, 200):
            self.buffer_combo.addItem(f"{ms} мс", ms)
        self.buffer_combo.currentIndexChanged.connect(self._on_buffer)
        row.addWidget(self.buffer_combo)
        self.stats = QLabel("", objectName="hint")
        self.stats.setMinimumWidth(230)
        self.stats.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.stats)
        return row

    def _slider(self, key, title, low, high, formatter, step=1):
        layout = QVBoxLayout()
        layout.setSpacing(2)
        top = QHBoxLayout()
        top.addWidget(QLabel(title))
        top.addStretch(1)
        value = QLabel("", objectName="value")
        top.addWidget(value)
        layout.addLayout(top)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setSingleStep(step)
        slider.setPageStep(step * 5)

        def changed(raw):
            snapped = round(raw / step) * step
            value.setText(formatter(snapped))
            if not self._updating:
                setattr(self.settings, key, snapped)
                self._after_change()

        slider.valueChanged.connect(changed)
        self._sliders[key] = (slider, changed)
        layout.addWidget(slider)
        return layout

    # ---- state ----

    def _fill_devices(self):
        self._updating = True
        settings = self.settings
        inputs, outputs = audio.list_devices("input"), audio.list_devices("output")
        if settings.input_device not in inputs:
            settings.input_device = audio.default_input()
        if settings.output_device not in outputs:
            settings.output_device = audio.default_output()
        real_outputs = [n for n in outputs if not any(s in n for s in audio.SILENT_OUTPUTS)]
        if settings.monitor_device not in real_outputs:
            settings.monitor_device = real_outputs[0] if real_outputs else ""
        for combo, names, current in ((self.input_combo, inputs, settings.input_device),
                                      (self.output_combo, outputs, settings.output_device),
                                      (self.monitor_combo, real_outputs, settings.monitor_device)):
            combo.clear()
            combo.addItems(names)
            combo.setCurrentText(current)
        self._updating = False
        self._update_obs_hint()

    def _sync_controls(self):
        self._updating = True
        settings = self.settings
        for key, (slider, changed) in self._sliders.items():
            slider.setValue(getattr(settings, key))
            changed(getattr(settings, key))
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(settings.engine)))
        self.buffer_combo.setCurrentIndex(max(0, self.buffer_combo.findData(settings.buffer_ms)))
        self.power_button.setChecked(settings.enabled)
        self.power_button.setText("Шумодав включён" if settings.enabled else "Шумодав выключен")
        self.mute_button.setChecked(settings.muted)
        self.mute_button.setText("Микрофон выключен" if settings.muted else "Выключить микрофон")
        self._updating = False
        self.input_meter.set_mark(settings.gate_db if settings.gate_db > -80 else None)

    def _update_obs_hint(self):
        name = self.settings.output_device
        if audio.is_cable(name):
            text = "В OBS: «Захват входного аудио» → CABLE Output."
        elif "Steam Streaming Speakers" in name:
            text = "В OBS: «Захват звука приложения» → окно AIVoice. Голос не попадёт в ваши колонки."
        elif name:
            text = "В OBS: «Захват звука приложения» → окно AIVoice. Голос будет слышен и на этом устройстве."
        else:
            text = "Нет подходящего выхода. Установите VB-Cable или выберите устройство вручную."
        self.obs_hint.setText(text)

    def restart_audio(self):
        self.engine.start()
        self._restart_wait = 0
        self._apply_monitor()

    def _after_change(self):
        self.input_meter.set_mark(self.settings.gate_db if self.settings.gate_db > -80 else None)
        self._save_timer.start()

    def _save(self):
        try:
            store.save(self.settings)
        except OSError as exc:
            self.status.setText(f"Не удалось сохранить настройки: {exc}")

    # ---- events ----

    def _on_power(self, checked):
        if self._updating:
            return
        self.settings.enabled = checked
        self.power_button.setText("Шумодав включён" if checked else "Шумодав выключен")
        self._after_change()

    def _on_mute(self, checked):
        if self._updating:
            return
        self.settings.muted = checked
        self.mute_button.setText("Микрофон выключен" if checked else "Выключить микрофон")
        self._after_change()

    def _on_device(self, key, name):
        if self._updating or not name:
            return
        setattr(self.settings, key, name)
        self._update_obs_hint()
        self._after_change()
        self.restart_audio()

    def _on_engine(self, index):
        if self._updating:
            return
        self.settings.engine = self.engine_combo.itemData(index)
        self._after_change()
        self.restart_audio()

    def _on_buffer(self, index):
        if self._updating:
            return
        self.settings.buffer_ms = self.buffer_combo.itemData(index)
        self._after_change()
        self.restart_audio()

    def _on_monitor_device(self, name):
        if self._updating or not name:
            return
        self.settings.monitor_device = name
        self._after_change()
        self._apply_monitor()

    def _apply_monitor(self, *_):
        name = self.settings.monitor_device if self.monitor_check.isChecked() else ""
        try:
            self.engine.set_monitor(name)
        except Exception as exc:
            self.engine.error = f"Не удалось открыть «{name}»: {exc}"

    def _on_refresh(self):
        self.engine.stop()
        audio.refresh_devices()
        self._fill_devices()
        self._after_change()
        self.restart_audio()

    def _on_reset(self):
        defaults = store.Settings()
        for key in ("strength", "highpass_hz", "gate_db", "compression", "input_gain_db", "output_gain_db"):
            setattr(self.settings, key, getattr(defaults, key))
        self._sync_controls()
        self._after_change()

    def _tick(self):
        engine = self.engine
        pipeline = engine.pipeline if engine.running else None
        self.input_meter.set_level(pipeline.input_db if pipeline else METER_FLOOR_DB)
        self.output_meter.set_level(pipeline.output_db if pipeline else METER_FLOOR_DB)

        if engine.needs_restart or (not engine.running and self.settings.input_device and self.settings.output_device):
            # A device was unplugged or the system took it away: keep retrying until it returns.
            self._restart_wait += self._timer.interval()
            if self._restart_wait >= RESTART_INTERVAL_MS:
                self.engine.stop()
                audio.refresh_devices()
                self.restart_audio()

        if engine.running and not engine.needs_restart:
            self._set_status("  Работает", "hint")
            self.stats.setText(f"Нагрузка {engine.load * 100:.0f}% · задержка ~{engine.latency_ms:.0f} мс · сбои {engine.underruns + engine.dropped}")
        else:
            self._set_status("  " + (engine.error or "Устройство недоступно, пробую подключиться…"), "error")
            self.stats.setText("")

    def _set_status(self, text, style_name):
        self.status.setText(text)
        if self.status.objectName() != style_name:
            self.status.setObjectName(style_name)
            self.status.style().polish(self.status)

    def closeEvent(self, event):
        self._timer.stop()
        self.engine.stop()
        self._save()
        super().closeEvent(event)


def _make_icon():
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#3d6df2"))
    painter.drawRoundedRect(0, 0, 64, 64, 14, 14)
    painter.setBrush(QColor("#ffffff"))
    for x, height in ((12, 16), (24, 36), (36, 44), (48, 22)):
        painter.drawRoundedRect(QRectF(x - 3, 32 - height / 2, 6, height), 3, 3)
    painter.end()
    return QIcon(pixmap)


def build_app(argv):
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(STYLE)
    return app
