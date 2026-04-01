#!/usr/bin/env python3
"""EEG Cursor — Comprehensive Graphical User Interface.

PyQt5 GUI that provides:
  - Dashboard with real-time EEG signal display, classification bars, cursor indicator
  - Data collection controls (calibration, ERP trainer, JEPA pre-training)
  - Model training with type selection, data browsing, cross-validation display
  - Live control with toggle switches for adaptation, state monitoring, auto-undo
  - Settings panel for all configuration parameters
  - Scrolling coloured log output
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QGroupBox, QTextEdit, QComboBox,
        QProgressBar, QStatusBar, QSplitter, QFrame, QFileDialog,
        QMessageBox, QTabWidget, QGridLayout, QSpinBox, QDoubleSpinBox,
        QCheckBox, QLineEdit, QScrollArea, QSizePolicy,
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
    from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPainter, QBrush, QPen
except ImportError:
    print("PyQt5 is required for the GUI.")
    print("Install with: pip install PyQt5")
    sys.exit(1)

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

from src.config import load_config

logger = logging.getLogger("gui")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = str(Path(__file__).parent.parent)
CLASS_NAMES = ["rest", "left_hand", "right_hand", "feet", "tongue"]
CLASS_COLORS = {
    "rest":       "#888888",
    "left_hand":  "#4ec9b0",
    "right_hand": "#569cd6",
    "feet":       "#dcdcaa",
    "tongue":     "#ce9178",
}
CHANNEL_NAMES_16 = [
    "C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "F3",
    "F4", "P3", "P4", "O1", "O2", "P3p", "FCz", "Fz",
]

# Dark theme stylesheet — shared across the whole application
DARK_STYLE = """
    QMainWindow { background-color: #1e1e1e; }
    QWidget { background-color: #1e1e1e; color: #cccccc; }
    QGroupBox {
        border: 1px solid #444;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 16px;
        font-weight: bold;
        color: #aaa;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    QPushButton {
        background-color: #2d2d2d;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 8px 16px;
        color: #ccc;
        font-size: 13px;
        min-height: 32px;
    }
    QPushButton:hover { background-color: #3d3d3d; border-color: #0078d4; }
    QPushButton:pressed { background-color: #0078d4; }
    QPushButton:disabled { background-color: #1a1a1a; color: #555; }
    QPushButton#primary {
        background-color: #0078d4;
        border-color: #0078d4;
        color: white;
        font-weight: bold;
    }
    QPushButton#primary:hover { background-color: #1a8ae8; }
    QPushButton#danger {
        background-color: #d42020;
        border-color: #d42020;
        color: white;
    }
    QPushButton#danger:hover { background-color: #e83030; }
    QPushButton#success {
        background-color: #1a8a1a;
        border-color: #1a8a1a;
        color: white;
        font-weight: bold;
    }
    QPushButton#success:hover { background-color: #2ab02a; }
    QTextEdit {
        background-color: #0d0d0d;
        border: 1px solid #333;
        border-radius: 4px;
        color: #ccc;
        font-family: 'Cascadia Code', 'Consolas', monospace;
        font-size: 12px;
    }
    QLabel { color: #ccc; }
    QLabel#status-good { color: #4ec94e; }
    QLabel#status-bad { color: #ff6666; }
    QLabel#status-warn { color: #ffaa00; }
    QLabel#section-header { color: #0078d4; font-size: 14px; font-weight: bold; }
    QComboBox {
        background-color: #2d2d2d;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 4px 8px;
        color: #ccc;
    }
    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        color: #ccc;
        selection-background-color: #0078d4;
    }
    QTabWidget::pane { border: 1px solid #444; }
    QTabBar::tab {
        background-color: #2d2d2d;
        border: 1px solid #444;
        padding: 8px 16px;
        color: #aaa;
        min-width: 100px;
    }
    QTabBar::tab:selected {
        background-color: #1e1e1e;
        color: #fff;
        border-bottom: 2px solid #0078d4;
    }
    QTabBar::tab:hover { color: #fff; }
    QProgressBar {
        background-color: #2d2d2d;
        border: 1px solid #444;
        border-radius: 4px;
        text-align: center;
        color: #ccc;
        font-size: 11px;
    }
    QProgressBar::chunk {
        background-color: #0078d4;
        border-radius: 3px;
    }
    QCheckBox { color: #ccc; spacing: 8px; }
    QCheckBox::indicator {
        width: 18px; height: 18px;
        border: 1px solid #555;
        border-radius: 3px;
        background-color: #2d2d2d;
    }
    QCheckBox::indicator:checked {
        background-color: #0078d4;
        border-color: #0078d4;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox {
        background-color: #2d2d2d;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 4px 8px;
        color: #ccc;
    }
    QScrollArea { border: none; }
    QScrollBar:vertical {
        background: #1e1e1e; width: 10px; border: none;
    }
    QScrollBar::handle:vertical {
        background: #555; border-radius: 5px; min-height: 30px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


# ---------------------------------------------------------------------------
# Log handler that emits to the GUI
# ---------------------------------------------------------------------------
class QTextEditHandler(logging.Handler):
    """Logging handler that writes coloured HTML to a QTextEdit widget."""

    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self._text_edit = text_edit

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        color = {
            "DEBUG":    "#888888",
            "INFO":     "#CCCCCC",
            "WARNING":  "#FFAA00",
            "ERROR":    "#FF4444",
            "CRITICAL": "#FF0000",
        }.get(record.levelname, "#CCCCCC")
        self._text_edit.append(f'<span style="color:{color}">{msg}</span>')
        # Auto-scroll to bottom
        sb = self._text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())


# ---------------------------------------------------------------------------
# Worker thread for long-running subprocess tasks
# ---------------------------------------------------------------------------
class WorkerThread(QThread):
    """Run a subprocess command and emit output line-by-line."""

    output = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, cmd: list, cwd: str):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self._process: Optional[subprocess.Popen] = None

    def run(self):
        try:
            self._process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.cwd,
                bufsize=1,
            )
            for line in self._process.stdout:
                self.output.emit(line.rstrip())
            self._process.wait()
            self.finished_signal.emit(self._process.returncode)
        except Exception as e:
            self.output.emit(f"ERROR: {e}")
            self.finished_signal.emit(1)

    def stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()


# ---------------------------------------------------------------------------
# EEG Signal Display (pyqtgraph)
# ---------------------------------------------------------------------------
class SignalDisplay(QWidget):
    """Real-time EEG signal display using pyqtgraph (up to 8 channels)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_PYQTGRAPH:
            self._plot_widget = pg.GraphicsLayoutWidget()
            self._plot_widget.setBackground("#1e1e1e")
            layout.addWidget(self._plot_widget)
            self._plots = []
            self._curves = []
            self._initialized = False
        else:
            label = QLabel("pyqtgraph not installed.\nInstall with: pip install pyqtgraph")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: #888; font-size: 14px;")
            layout.addWidget(label)
            self._initialized = False

    def init_channels(self, n_channels: int = 8, channel_names: list = None):
        if not HAS_PYQTGRAPH:
            return
        self._plot_widget.clear()
        self._plots = []
        self._curves = []
        colors = [
            (100, 200, 255), (255, 150, 100), (100, 255, 150),
            (255, 255, 100), (200, 100, 255), (255, 100, 200),
            (100, 255, 255), (200, 200, 200),
        ]
        n = min(n_channels, 8)
        for i in range(n):
            name = channel_names[i] if channel_names and i < len(channel_names) else f"Ch {i}"
            p = self._plot_widget.addPlot(row=i, col=0)
            p.setLabel("left", name)
            p.showAxis("bottom", i == n - 1)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=False)
            p.setYRange(-100, 100)
            p.hideButtons()
            curve = p.plot(pen=pg.mkPen(color=colors[i % len(colors)], width=1))
            self._plots.append(p)
            self._curves.append(curve)
        self._initialized = True

    def update_data(self, data: np.ndarray):
        if not self._initialized or not HAS_PYQTGRAPH:
            return
        for i, curve in enumerate(self._curves):
            if i < data.shape[0]:
                curve.setData(data[i])


# ---------------------------------------------------------------------------
# Classification Bar Display
# ---------------------------------------------------------------------------
class ClassificationBars(QWidget):
    """Horizontal probability bars for each class."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self._bars = {}
        self._labels = {}
        for cls in CLASS_NAMES:
            row = QHBoxLayout()
            lbl = QLabel(cls.replace("_", " ").title())
            lbl.setFixedWidth(90)
            lbl.setStyleSheet("font-size: 11px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(18)
            bar.setStyleSheet(f"""
                QProgressBar {{ background-color: #2d2d2d; border: 1px solid #444; border-radius: 3px; }}
                QProgressBar::chunk {{ background-color: {CLASS_COLORS.get(cls, '#0078d4')}; border-radius: 2px; }}
            """)
            val_lbl = QLabel("0%")
            val_lbl.setFixedWidth(40)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet("font-size: 11px; color: #aaa;")
            row.addWidget(lbl)
            row.addWidget(bar, stretch=1)
            row.addWidget(val_lbl)
            layout.addLayout(row)
            self._bars[cls] = bar
            self._labels[cls] = val_lbl

    def update_probabilities(self, probs: dict):
        """Update bars from a {class_name: probability} dict."""
        for cls, bar in self._bars.items():
            p = probs.get(cls, 0.0)
            pct = int(p * 100)
            bar.setValue(pct)
            self._labels[cls].setText(f"{pct}%")


# ---------------------------------------------------------------------------
# Cursor Position Indicator
# ---------------------------------------------------------------------------
class CursorIndicator(QWidget):
    """Small 2D canvas showing cursor position."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(120, 120)
        self.setMaximumSize(160, 160)
        self._x = 0.5
        self._y = 0.5

    def set_position(self, x: float, y: float):
        self._x = max(0.0, min(1.0, x))
        self._y = max(0.0, min(1.0, y))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # Background
        painter.fillRect(0, 0, w, h, QColor("#1a1a1a"))
        # Grid
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.drawLine(w // 2, 0, w // 2, h)
        painter.drawLine(0, h // 2, w, h // 2)
        # Border
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)
        # Cursor dot
        cx = int(self._x * w)
        cy = int(self._y * h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#0078d4")))
        painter.drawEllipse(cx - 6, cy - 6, 12, 12)
        # Inner dot
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawEllipse(cx - 2, cy - 2, 4, 4)
        painter.end()


# ---------------------------------------------------------------------------
# Status LED indicator
# ---------------------------------------------------------------------------
class StatusLED(QWidget):
    """Small coloured circle indicator."""

    def __init__(self, color: str = "#ff6666", parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self._color = color

    def set_color(self, color: str):
        self._color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(self._color)))
        painter.drawEllipse(1, 1, 12, 12)
        painter.end()


# ---------------------------------------------------------------------------
# Toggle Switch (styled checkbox)
# ---------------------------------------------------------------------------
def make_toggle(label_text: str, checked: bool = False) -> tuple:
    """Create a labelled checkbox toggle. Returns (layout, checkbox)."""
    row = QHBoxLayout()
    cb = QCheckBox(label_text)
    cb.setChecked(checked)
    cb.setStyleSheet("QCheckBox { font-size: 13px; }")
    row.addWidget(cb)
    row.addStretch()
    return row, cb


# ---------------------------------------------------------------------------
# TAB 1: Dashboard
# ---------------------------------------------------------------------------
class DashboardTab(QWidget):
    """System overview with signals, classification bars, cursor, stats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- Top row: status indicators ---
        status_group = QGroupBox("System Status")
        sg = QGridLayout(status_group)
        self.led_board = StatusLED("#ff6666")
        self.led_model = StatusLED("#ff6666")
        self.led_gpu = StatusLED("#ffaa00")
        self.led_adapt = StatusLED("#888888")
        self.lbl_board = QLabel("Disconnected")
        self.lbl_model = QLabel("No model")
        self.lbl_gpu = QLabel("Checking...")
        self.lbl_adapt = QLabel("Disabled")
        for i, (led, name_lbl, txt_lbl) in enumerate([
            (self.led_board, QLabel("Board:"), self.lbl_board),
            (self.led_model, QLabel("Model:"), self.lbl_model),
            (self.led_gpu, QLabel("GPU:"), self.lbl_gpu),
            (self.led_adapt, QLabel("Adaptation:"), self.lbl_adapt),
        ]):
            sg.addWidget(led, i, 0)
            sg.addWidget(name_lbl, i, 1)
            sg.addWidget(txt_lbl, i, 2)
        sg.setColumnStretch(2, 1)
        layout.addWidget(status_group)

        # --- Middle: signals + classification + cursor ---
        mid_splitter = QSplitter(Qt.Horizontal)

        # EEG signals (left, 60%)
        self.signal_display = SignalDisplay()
        self.signal_display.init_channels(8, CHANNEL_NAMES_16[:8])
        mid_splitter.addWidget(self.signal_display)

        # Right panel: classification bars + cursor + stats
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 0, 0, 0)

        cls_group = QGroupBox("Classification Output")
        cls_layout = QVBoxLayout(cls_group)
        self.class_bars = ClassificationBars()
        cls_layout.addWidget(self.class_bars)
        rl.addWidget(cls_group)

        cursor_group = QGroupBox("Cursor Position")
        cur_layout = QHBoxLayout(cursor_group)
        self.cursor_indicator = CursorIndicator()
        cur_layout.addStretch()
        cur_layout.addWidget(self.cursor_indicator)
        cur_layout.addStretch()
        rl.addWidget(cursor_group)

        stats_group = QGroupBox("Session Statistics")
        stats_grid = QGridLayout(stats_group)
        self.lbl_cls_rate = QLabel("0.0")
        self.lbl_accuracy = QLabel("--")
        self.lbl_latency = QLabel("--")
        self.lbl_session_time = QLabel("00:00")
        stats_grid.addWidget(QLabel("Classifications/s:"), 0, 0)
        stats_grid.addWidget(self.lbl_cls_rate, 0, 1)
        stats_grid.addWidget(QLabel("Accuracy:"), 1, 0)
        stats_grid.addWidget(self.lbl_accuracy, 1, 1)
        stats_grid.addWidget(QLabel("Latency (ms):"), 2, 0)
        stats_grid.addWidget(self.lbl_latency, 2, 1)
        stats_grid.addWidget(QLabel("Session Time:"), 3, 0)
        stats_grid.addWidget(self.lbl_session_time, 3, 1)
        stats_grid.setColumnStretch(1, 1)
        rl.addWidget(stats_group)

        rl.addStretch()
        mid_splitter.addWidget(right)
        mid_splitter.setSizes([600, 300])
        layout.addWidget(mid_splitter, stretch=1)

    def refresh_status(self, config: dict):
        """Update status LEDs and labels based on project state."""
        models_dir = Path(PROJECT_DIR) / "models"
        data_dir = Path(PROJECT_DIR) / "data" / "raw"

        # Model
        models = sorted(models_dir.glob("*.pkl")) + sorted(models_dir.glob("*.pt"))
        if models:
            self.lbl_model.setText(models[-1].name)
            self.led_model.set_color("#4ec94e")
        else:
            self.lbl_model.setText("No model")
            self.led_model.set_color("#ff6666")

        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.lbl_gpu.setText(torch.cuda.get_device_name(0))
                self.led_gpu.set_color("#4ec94e")
            else:
                self.lbl_gpu.setText("CPU only")
                self.led_gpu.set_color("#ffaa00")
        except ImportError:
            self.lbl_gpu.setText("PyTorch N/A")
            self.led_gpu.set_color("#ff6666")

        # Adaptation
        adapt_on = config.get("adaptation", {}).get("enabled", False)
        self.lbl_adapt.setText("Enabled" if adapt_on else "Disabled")
        self.led_adapt.set_color("#4ec94e" if adapt_on else "#888888")


# ---------------------------------------------------------------------------
# TAB 2: Data Collection
# ---------------------------------------------------------------------------
class DataCollectionTab(QWidget):
    """Calibration, ERP trainer, JEPA pre-training controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Calibration
        cal_group = QGroupBox("Motor Imagery Calibration (Graz Paradigm)")
        cal_layout = QVBoxLayout(cal_group)
        cal_desc = QLabel(
            "Run the standard Graz calibration paradigm to collect labelled\n"
            "motor imagery trials. Presents visual cues and records EEG."
        )
        cal_desc.setStyleSheet("color: #999; font-size: 12px;")
        cal_layout.addWidget(cal_desc)
        self.btn_calibrate = QPushButton("Start Calibration")
        self.btn_calibrate.setObjectName("primary")
        self.btn_calibrate.setMinimumHeight(44)
        cal_layout.addWidget(self.btn_calibrate)
        layout.addWidget(cal_group)

        # ERP Trainer
        erp_group = QGroupBox("ERP Signal Trainer")
        erp_layout = QVBoxLayout(erp_group)
        erp_desc = QLabel(
            "Real-time ERP/ERDS feedback during motor imagery practice.\n"
            "Helps subjects learn to produce consistent, detectable signals."
        )
        erp_desc.setStyleSheet("color: #999; font-size: 12px;")
        erp_layout.addWidget(erp_desc)
        self.btn_erp = QPushButton("ERP Signal Trainer")
        self.btn_erp.setMinimumHeight(40)
        self.btn_erp.setToolTip(
            "ERP data is for exploration only — use Calibration for training data."
        )
        erp_layout.addWidget(self.btn_erp)
        layout.addWidget(erp_group)

        # JEPA Pre-Training
        jepa_group = QGroupBox("JEPA Self-Supervised Pre-Training")
        jepa_layout = QVBoxLayout(jepa_group)
        jepa_desc = QLabel(
            "Pre-train encoder on unlabelled EEG data using Joint Embedding\n"
            "Predictive Architecture. Improves downstream classification."
        )
        jepa_desc.setStyleSheet("color: #999; font-size: 12px;")
        jepa_layout.addWidget(jepa_desc)
        self.btn_jepa = QPushButton("JEPA Pre-Training")
        self.btn_jepa.setMinimumHeight(40)
        jepa_layout.addWidget(self.btn_jepa)
        layout.addWidget(jepa_group)

        # Recording status
        rec_group = QGroupBox("Recording Status")
        rec_layout = QGridLayout(rec_group)
        self.lbl_rec_status = QLabel("Idle")
        self.lbl_rec_status.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.lbl_trial_count = QLabel("0")
        self.lbl_last_file = QLabel("None")
        self.lbl_total_recordings = QLabel("0")
        rec_layout.addWidget(QLabel("Status:"), 0, 0)
        rec_layout.addWidget(self.lbl_rec_status, 0, 1)
        rec_layout.addWidget(QLabel("Trial Count:"), 1, 0)
        rec_layout.addWidget(self.lbl_trial_count, 1, 1)
        rec_layout.addWidget(QLabel("Last File:"), 2, 0)
        rec_layout.addWidget(self.lbl_last_file, 2, 1)
        rec_layout.addWidget(QLabel("Total Recordings:"), 3, 0)
        rec_layout.addWidget(self.lbl_total_recordings, 3, 1)
        rec_layout.setColumnStretch(1, 1)
        layout.addWidget(rec_group)

        layout.addStretch()

    def refresh_file_info(self):
        """Update recording file information."""
        data_dir = Path(PROJECT_DIR) / "data" / "raw"
        recordings = sorted(data_dir.glob("*.npz"))
        self.lbl_total_recordings.setText(str(len(recordings)))
        if recordings:
            last = recordings[-1]
            size_kb = last.stat().st_size / 1024
            self.lbl_last_file.setText(f"{last.name} ({size_kb:.0f} KB)")
        else:
            self.lbl_last_file.setText("None")


# ---------------------------------------------------------------------------
# TAB 3: Training
# ---------------------------------------------------------------------------
class TrainingTab(QWidget):
    """Model training controls with type selection and results display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Model type selector
        type_group = QGroupBox("Model Configuration")
        type_layout = QGridLayout(type_group)
        type_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "csp_lda", "eegnet", "riemannian", "neural_sde", "adaptive_router",
        ])
        type_layout.addWidget(self.combo_model, 0, 1)

        type_layout.addWidget(QLabel("Data File:"), 1, 0)
        data_row = QHBoxLayout()
        self.txt_data_path = QLineEdit()
        self.txt_data_path.setPlaceholderText("Select .npz recording file...")
        self.txt_data_path.setReadOnly(True)
        data_row.addWidget(self.txt_data_path)
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.setFixedWidth(90)
        self.btn_browse.clicked.connect(self._browse_data)
        data_row.addWidget(self.btn_browse)
        type_layout.addLayout(data_row, 1, 1)

        self.btn_use_latest = QPushButton("Use Latest Recording")
        self.btn_use_latest.clicked.connect(self._use_latest)
        type_layout.addWidget(self.btn_use_latest, 2, 1)

        type_layout.setColumnStretch(1, 1)
        layout.addWidget(type_group)

        # Train button
        self.btn_train = QPushButton("Train Model")
        self.btn_train.setObjectName("primary")
        self.btn_train.setMinimumHeight(44)
        self.btn_train.setFont(QFont("Segoe UI", 13, QFont.Bold))
        layout.addWidget(self.btn_train)

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Cross-validation results
        cv_group = QGroupBox("Cross-Validation Results")
        cv_layout = QVBoxLayout(cv_group)
        self.txt_cv_results = QTextEdit()
        self.txt_cv_results.setReadOnly(True)
        self.txt_cv_results.setMaximumHeight(180)
        self.txt_cv_results.setPlaceholderText("Train a model to see results here...")
        cv_layout.addWidget(self.txt_cv_results)
        layout.addWidget(cv_group)

        # Trained model info
        model_group = QGroupBox("Trained Model Info")
        model_layout = QGridLayout(model_group)
        self.lbl_model_name = QLabel("None")
        self.lbl_model_type = QLabel("--")
        self.lbl_model_acc = QLabel("--")
        self.lbl_model_date = QLabel("--")
        model_layout.addWidget(QLabel("Model File:"), 0, 0)
        model_layout.addWidget(self.lbl_model_name, 0, 1)
        model_layout.addWidget(QLabel("Type:"), 1, 0)
        model_layout.addWidget(self.lbl_model_type, 1, 1)
        model_layout.addWidget(QLabel("CV Accuracy:"), 2, 0)
        model_layout.addWidget(self.lbl_model_acc, 2, 1)
        model_layout.addWidget(QLabel("Trained:"), 3, 0)
        model_layout.addWidget(self.lbl_model_date, 3, 1)
        model_layout.setColumnStretch(1, 1)
        layout.addWidget(model_group)

        layout.addStretch()

    def _browse_data(self):
        data_dir = str(Path(PROJECT_DIR) / "data" / "raw")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Data", data_dir, "NumPy Files (*.npz);;All Files (*)"
        )
        if path:
            self.txt_data_path.setText(path)

    def _use_latest(self):
        data_dir = Path(PROJECT_DIR) / "data" / "raw"
        recordings = sorted(data_dir.glob("*.npz"))
        if recordings:
            self.txt_data_path.setText(str(recordings[-1]))
        else:
            QMessageBox.warning(self, "No Data", "No recordings found in data/raw/.")

    def get_data_path(self) -> Optional[str]:
        path = self.txt_data_path.text().strip()
        return path if path else None

    def refresh_model_info(self):
        """Update trained model information."""
        models_dir = Path(PROJECT_DIR) / "models"
        models = sorted(
            list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.pt"))
        )
        if models:
            latest = models[-1]
            self.lbl_model_name.setText(latest.name)
            # Guess type from name
            name = latest.stem.lower()
            for t in ["csp_lda", "eegnet", "riemannian", "neural_sde", "adaptive_router"]:
                if t.replace("_", "") in name.replace("_", ""):
                    self.lbl_model_type.setText(t)
                    break
            import datetime as dt
            mtime = dt.datetime.fromtimestamp(latest.stat().st_mtime)
            self.lbl_model_date.setText(mtime.strftime("%Y-%m-%d %H:%M"))


# ---------------------------------------------------------------------------
# TAB 4: Live Control
# ---------------------------------------------------------------------------
class LiveControlTab(QWidget):
    """Start/stop live EEG cursor with toggle switches for features."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Big start/stop buttons
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("START")
        self.btn_start.setObjectName("success")
        self.btn_start.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.btn_start.setMinimumHeight(70)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.btn_stop.setMinimumHeight(70)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

        # Feature toggles
        toggle_group = QGroupBox("Feature Toggles")
        tg_layout = QVBoxLayout(toggle_group)

        row1, self.chk_seal = make_toggle("Self-Adaptation (SEAL)", True)
        tg_layout.addLayout(row1)
        row2, self.chk_state_mon = make_toggle("State Monitoring (fatigue/attention)", False)
        tg_layout.addLayout(row2)
        row3, self.chk_auto_undo = make_toggle("Auto-Undo on ErrP Detection", True)
        tg_layout.addLayout(row3)
        row4, self.chk_adaptive_routing = make_toggle("Adaptive Classifier Routing", False)
        tg_layout.addLayout(row4)

        layout.addWidget(toggle_group)

        # Live status
        status_group = QGroupBox("Live Status")
        status_layout = QGridLayout(status_group)
        self.lbl_current_class = QLabel("--")
        self.lbl_current_class.setStyleSheet("font-size: 18px; font-weight: bold; color: #0078d4;")
        self.lbl_direction = QLabel("--")
        self.lbl_velocity = QLabel("0.0")
        self.lbl_clicks = QLabel("0")
        self.lbl_errp_count = QLabel("0")
        self.lbl_p300_count = QLabel("0")
        status_layout.addWidget(QLabel("Current Class:"), 0, 0)
        status_layout.addWidget(self.lbl_current_class, 0, 1)
        status_layout.addWidget(QLabel("Direction:"), 1, 0)
        status_layout.addWidget(self.lbl_direction, 1, 1)
        status_layout.addWidget(QLabel("Velocity:"), 2, 0)
        status_layout.addWidget(self.lbl_velocity, 2, 1)
        status_layout.addWidget(QLabel("Clicks:"), 3, 0)
        status_layout.addWidget(self.lbl_clicks, 3, 1)
        status_layout.addWidget(QLabel("ErrP Detections:"), 0, 2)
        status_layout.addWidget(self.lbl_errp_count, 0, 3)
        status_layout.addWidget(QLabel("P300 Detections:"), 1, 2)
        status_layout.addWidget(self.lbl_p300_count, 1, 3)
        status_layout.setColumnStretch(1, 1)
        status_layout.setColumnStretch(3, 1)
        layout.addWidget(status_group)

        # ErrP / P300 detection log
        erp_group = QGroupBox("ErrP / P300 Detection Log")
        erp_layout = QVBoxLayout(erp_group)
        self.txt_erp_log = QTextEdit()
        self.txt_erp_log.setReadOnly(True)
        self.txt_erp_log.setMaximumHeight(160)
        self.txt_erp_log.setPlaceholderText("ErrP and P300 detections will appear here...")
        erp_layout.addWidget(self.txt_erp_log)
        layout.addWidget(erp_group)

        layout.addStretch()

    def set_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.chk_seal.setEnabled(not running)
        self.chk_state_mon.setEnabled(not running)
        self.chk_auto_undo.setEnabled(not running)
        self.chk_adaptive_routing.setEnabled(not running)


# ---------------------------------------------------------------------------
# TAB 5: Settings
# ---------------------------------------------------------------------------
class SettingsTab(QWidget):
    """All configuration parameters in a scrollable form."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        # -- Board --
        board_group = QGroupBox("Board Configuration")
        bg = QGridLayout(board_group)
        board_cfg = config.get("board", {})
        bg.addWidget(QLabel("Board ID:"), 0, 0)
        self.spin_board_id = QSpinBox()
        self.spin_board_id.setRange(-1, 50)
        self.spin_board_id.setValue(board_cfg.get("board_id", -1))
        bg.addWidget(self.spin_board_id, 0, 1)
        bg.addWidget(QLabel("-1=Synthetic, 0=Cyton, 2=Cyton+Daisy"), 0, 2)

        bg.addWidget(QLabel("Serial Port:"), 1, 0)
        self.txt_serial = QLineEdit(board_cfg.get("serial_port", ""))
        self.txt_serial.setPlaceholderText("e.g. COM3 or /dev/ttyUSB0")
        bg.addWidget(self.txt_serial, 1, 1, 1, 2)

        bg.addWidget(QLabel("Channel Count:"), 2, 0)
        self.spin_channels = QSpinBox()
        self.spin_channels.setRange(1, 32)
        self.spin_channels.setValue(board_cfg.get("channel_count", 16))
        bg.addWidget(self.spin_channels, 2, 1)
        bg.setColumnStretch(1, 1)
        layout.addWidget(board_group)

        # -- Preprocessing --
        pre_group = QGroupBox("Preprocessing")
        pg_layout = QGridLayout(pre_group)
        pre_cfg = config.get("preprocessing", {})

        pg_layout.addWidget(QLabel("Bandpass Low (Hz):"), 0, 0)
        self.dspin_bp_low = QDoubleSpinBox()
        self.dspin_bp_low.setRange(0.1, 50.0)
        self.dspin_bp_low.setValue(pre_cfg.get("bandpass_low", 1.0))
        self.dspin_bp_low.setSingleStep(0.5)
        pg_layout.addWidget(self.dspin_bp_low, 0, 1)

        pg_layout.addWidget(QLabel("Bandpass High (Hz):"), 1, 0)
        self.dspin_bp_high = QDoubleSpinBox()
        self.dspin_bp_high.setRange(1.0, 100.0)
        self.dspin_bp_high.setValue(pre_cfg.get("bandpass_high", 40.0))
        self.dspin_bp_high.setSingleStep(1.0)
        pg_layout.addWidget(self.dspin_bp_high, 1, 1)

        pg_layout.addWidget(QLabel("Notch Freq (Hz):"), 2, 0)
        self.dspin_notch = QDoubleSpinBox()
        self.dspin_notch.setRange(0.0, 100.0)
        self.dspin_notch.setValue(pre_cfg.get("notch_freq", 60.0))
        pg_layout.addWidget(self.dspin_notch, 2, 1)

        pg_layout.addWidget(QLabel("CAR Enabled:"), 3, 0)
        self.chk_car = QCheckBox()
        self.chk_car.setChecked(pre_cfg.get("car_enabled", True))
        pg_layout.addWidget(self.chk_car, 3, 1)

        pg_layout.addWidget(QLabel("Artifact Threshold (uV):"), 4, 0)
        self.dspin_artifact = QDoubleSpinBox()
        self.dspin_artifact.setRange(10.0, 500.0)
        self.dspin_artifact.setValue(pre_cfg.get("artifact_threshold_uv", 100.0))
        pg_layout.addWidget(self.dspin_artifact, 4, 1)
        pg_layout.setColumnStretch(1, 1)
        layout.addWidget(pre_group)

        # -- Classification Window --
        cls_group = QGroupBox("Classification Window")
        cg = QGridLayout(cls_group)
        train_cfg = config.get("training", {})
        cg.addWidget(QLabel("Window Start (s):"), 0, 0)
        self.dspin_win_start = QDoubleSpinBox()
        self.dspin_win_start.setRange(0.0, 10.0)
        self.dspin_win_start.setValue(train_cfg.get("classification_window_start", 1.5))
        self.dspin_win_start.setSingleStep(0.1)
        cg.addWidget(self.dspin_win_start, 0, 1)
        cg.addWidget(QLabel("Window End (s):"), 1, 0)
        self.dspin_win_end = QDoubleSpinBox()
        self.dspin_win_end.setRange(0.5, 10.0)
        self.dspin_win_end.setValue(train_cfg.get("classification_window_end", 4.0))
        self.dspin_win_end.setSingleStep(0.1)
        cg.addWidget(self.dspin_win_end, 1, 1)
        cg.setColumnStretch(1, 1)
        layout.addWidget(cls_group)

        # -- Control --
        ctrl_group = QGroupBox("Cursor Control")
        ctrl_layout = QGridLayout(ctrl_group)
        ctrl_cfg = config.get("control", {})

        ctrl_layout.addWidget(QLabel("Max Velocity (px):"), 0, 0)
        self.spin_velocity = QSpinBox()
        self.spin_velocity.setRange(1, 100)
        self.spin_velocity.setValue(ctrl_cfg.get("max_velocity", 25))
        ctrl_layout.addWidget(self.spin_velocity, 0, 1)

        ctrl_layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        self.dspin_conf = QDoubleSpinBox()
        self.dspin_conf.setRange(0.0, 1.0)
        self.dspin_conf.setValue(ctrl_cfg.get("confidence_threshold", 0.5))
        self.dspin_conf.setSingleStep(0.05)
        ctrl_layout.addWidget(self.dspin_conf, 1, 1)

        ctrl_layout.addWidget(QLabel("Dead Zone:"), 2, 0)
        self.dspin_deadzone = QDoubleSpinBox()
        self.dspin_deadzone.setRange(0.0, 0.5)
        self.dspin_deadzone.setValue(ctrl_cfg.get("dead_zone", 0.15))
        self.dspin_deadzone.setSingleStep(0.01)
        ctrl_layout.addWidget(self.dspin_deadzone, 2, 1)

        ctrl_layout.addWidget(QLabel("Smoothing Alpha:"), 3, 0)
        self.dspin_smooth = QDoubleSpinBox()
        self.dspin_smooth.setRange(0.0, 1.0)
        self.dspin_smooth.setValue(ctrl_cfg.get("smoothing_alpha", 0.3))
        self.dspin_smooth.setSingleStep(0.05)
        ctrl_layout.addWidget(self.dspin_smooth, 3, 1)

        ctrl_layout.addWidget(QLabel("Update Rate (Hz):"), 4, 0)
        self.spin_update_rate = QSpinBox()
        self.spin_update_rate.setRange(1, 60)
        self.spin_update_rate.setValue(ctrl_cfg.get("update_rate_hz", 16))
        ctrl_layout.addWidget(self.spin_update_rate, 4, 1)
        ctrl_layout.setColumnStretch(1, 1)
        layout.addWidget(ctrl_group)

        # -- Adaptation --
        adapt_group = QGroupBox("Adaptation (SEAL)")
        ag = QGridLayout(adapt_group)
        adapt_cfg = config.get("adaptation", {})

        ag.addWidget(QLabel("Enabled:"), 0, 0)
        self.chk_adapt_enabled = QCheckBox()
        self.chk_adapt_enabled.setChecked(adapt_cfg.get("enabled", True))
        ag.addWidget(self.chk_adapt_enabled, 0, 1)

        ag.addWidget(QLabel("ErrP Threshold (uV):"), 1, 0)
        self.dspin_errp = QDoubleSpinBox()
        self.dspin_errp.setRange(1.0, 50.0)
        self.dspin_errp.setValue(adapt_cfg.get("errp_threshold", 8.0))
        ag.addWidget(self.dspin_errp, 1, 1)

        ag.addWidget(QLabel("Update Interval (s):"), 2, 0)
        self.spin_update_interval = QSpinBox()
        self.spin_update_interval.setRange(5, 300)
        self.spin_update_interval.setValue(adapt_cfg.get("update_interval_s", 30))
        ag.addWidget(self.spin_update_interval, 2, 1)

        ag.addWidget(QLabel("Auto-Undo:"), 3, 0)
        self.chk_auto_undo = QCheckBox()
        self.chk_auto_undo.setChecked(adapt_cfg.get("auto_undo", True))
        ag.addWidget(self.chk_auto_undo, 3, 1)
        ag.setColumnStretch(1, 1)
        layout.addWidget(adapt_group)

        # -- Advanced SOTA Toggles --
        adv_group = QGroupBox("Advanced SOTA Features")
        adv_layout = QGridLayout(adv_group)
        adv_cfg = config.get("advanced", {})

        self.chk_adv_routing = QCheckBox("Adaptive Routing")
        self.chk_adv_routing.setChecked(adv_cfg.get("adaptive_routing", False))
        adv_layout.addWidget(self.chk_adv_routing, 0, 0)

        self.chk_adv_state = QCheckBox("State Monitor")
        self.chk_adv_state.setChecked(adv_cfg.get("state_monitor", False))
        adv_layout.addWidget(self.chk_adv_state, 0, 1)

        self.chk_adv_causal = QCheckBox("Causal Discovery")
        self.chk_adv_causal.setChecked(adv_cfg.get("causal_discovery", False))
        adv_layout.addWidget(self.chk_adv_causal, 1, 0)

        self.chk_adv_koopman = QCheckBox("Koopman Spectral")
        self.chk_adv_koopman.setChecked(adv_cfg.get("koopman_enabled", False))
        adv_layout.addWidget(self.chk_adv_koopman, 1, 1)

        self.chk_adv_jepa = QCheckBox("JEPA Pre-Training")
        self.chk_adv_jepa.setChecked(adv_cfg.get("pretrain_enabled", False))
        adv_layout.addWidget(self.chk_adv_jepa, 2, 0)

        self.chk_adv_sde = QCheckBox("Neural SDE")
        self.chk_adv_sde.setChecked(adv_cfg.get("neural_sde_enabled", False))
        adv_layout.addWidget(self.chk_adv_sde, 2, 1)

        self.chk_adv_gflownet = QCheckBox("GFlowNet Adaptation")
        self.chk_adv_gflownet.setChecked(adv_cfg.get("gflownet_enabled", False))
        adv_layout.addWidget(self.chk_adv_gflownet, 3, 0)

        layout.addWidget(adv_group)

        # Save / Load buttons
        btn_row = QHBoxLayout()
        self.btn_save_config = QPushButton("Save Configuration")
        self.btn_save_config.setObjectName("primary")
        self.btn_save_config.clicked.connect(self._save_config)
        btn_row.addWidget(self.btn_save_config)

        self.btn_load_config = QPushButton("Load Configuration")
        self.btn_load_config.clicked.connect(self._load_config)
        btn_row.addWidget(self.btn_load_config)

        self.btn_reset_defaults = QPushButton("Reset Defaults")
        self.btn_reset_defaults.clicked.connect(self._reset_defaults)
        btn_row.addWidget(self.btn_reset_defaults)
        layout.addLayout(btn_row)

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def collect_config(self) -> dict:
        """Gather current UI values into a config dict."""
        cfg = {
            "board": {
                "board_id": self.spin_board_id.value(),
                "serial_port": self.txt_serial.text().strip(),
                "sampling_rate_override": None,
                "channel_count": self.spin_channels.value(),
            },
            "preprocessing": {
                "bandpass_low": self.dspin_bp_low.value(),
                "bandpass_high": self.dspin_bp_high.value(),
                "bandpass_order": 4,
                "notch_freq": self.dspin_notch.value(),
                "notch_quality": 30.0,
                "car_enabled": self.chk_car.isChecked(),
                "artifact_threshold_uv": self.dspin_artifact.value(),
            },
            "training": {
                **self._config.get("training", {}),
                "classification_window_start": self.dspin_win_start.value(),
                "classification_window_end": self.dspin_win_end.value(),
            },
            "control": {
                **self._config.get("control", {}),
                "max_velocity": self.spin_velocity.value(),
                "confidence_threshold": self.dspin_conf.value(),
                "dead_zone": self.dspin_deadzone.value(),
                "smoothing_alpha": self.dspin_smooth.value(),
                "update_rate_hz": self.spin_update_rate.value(),
            },
            "adaptation": {
                **self._config.get("adaptation", {}),
                "enabled": self.chk_adapt_enabled.isChecked(),
                "errp_threshold": self.dspin_errp.value(),
                "update_interval_s": self.spin_update_interval.value(),
                "auto_undo": self.chk_auto_undo.isChecked(),
            },
            "advanced": {
                **self._config.get("advanced", {}),
                "adaptive_routing": self.chk_adv_routing.isChecked(),
                "state_monitor": self.chk_adv_state.isChecked(),
                "causal_discovery": self.chk_adv_causal.isChecked(),
                "koopman_enabled": self.chk_adv_koopman.isChecked(),
                "pretrain_enabled": self.chk_adv_jepa.isChecked(),
                "neural_sde_enabled": self.chk_adv_sde.isChecked(),
                "gflownet_enabled": self.chk_adv_gflownet.isChecked(),
            },
            "features": self._config.get("features", {}),
            "classification": self._config.get("classification", {}),
            "ui": self._config.get("ui", {}),
            "paths": self._config.get("paths", {}),
        }
        return cfg

    def _save_config(self):
        """Save current settings to settings.yaml."""
        try:
            import yaml
            cfg = self.collect_config()
            path = Path(PROJECT_DIR) / "config" / "settings.yaml"
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            logger.info("Configuration saved to %s", path)
            QMessageBox.information(self, "Saved", f"Configuration saved to:\n{path}")
        except Exception as e:
            logger.error("Failed to save config: %s", e)
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def _load_config(self):
        """Load settings from a YAML file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration",
            str(Path(PROJECT_DIR) / "config"),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            try:
                import yaml
                with open(path, "r") as f:
                    cfg = yaml.safe_load(f)
                self._apply_config(cfg)
                logger.info("Configuration loaded from %s", path)
            except Exception as e:
                logger.error("Failed to load config: %s", e)
                QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")

    def _apply_config(self, cfg: dict):
        """Apply a config dict to the UI widgets."""
        self._config = cfg
        b = cfg.get("board", {})
        self.spin_board_id.setValue(b.get("board_id", -1))
        self.txt_serial.setText(b.get("serial_port", ""))
        self.spin_channels.setValue(b.get("channel_count", 16))

        p = cfg.get("preprocessing", {})
        self.dspin_bp_low.setValue(p.get("bandpass_low", 1.0))
        self.dspin_bp_high.setValue(p.get("bandpass_high", 40.0))
        self.dspin_notch.setValue(p.get("notch_freq", 60.0))
        self.chk_car.setChecked(p.get("car_enabled", True))
        self.dspin_artifact.setValue(p.get("artifact_threshold_uv", 100.0))

        t = cfg.get("training", {})
        self.dspin_win_start.setValue(t.get("classification_window_start", 1.5))
        self.dspin_win_end.setValue(t.get("classification_window_end", 4.0))

        c = cfg.get("control", {})
        self.spin_velocity.setValue(c.get("max_velocity", 25))
        self.dspin_conf.setValue(c.get("confidence_threshold", 0.5))
        self.dspin_deadzone.setValue(c.get("dead_zone", 0.15))
        self.dspin_smooth.setValue(c.get("smoothing_alpha", 0.3))
        self.spin_update_rate.setValue(c.get("update_rate_hz", 16))

        a = cfg.get("adaptation", {})
        self.chk_adapt_enabled.setChecked(a.get("enabled", True))
        self.dspin_errp.setValue(a.get("errp_threshold", 8.0))
        self.spin_update_interval.setValue(a.get("update_interval_s", 30))
        self.chk_auto_undo.setChecked(a.get("auto_undo", True))

        adv = cfg.get("advanced", {})
        self.chk_adv_routing.setChecked(adv.get("adaptive_routing", False))
        self.chk_adv_state.setChecked(adv.get("state_monitor", False))
        self.chk_adv_causal.setChecked(adv.get("causal_discovery", False))
        self.chk_adv_koopman.setChecked(adv.get("koopman_enabled", False))
        self.chk_adv_jepa.setChecked(adv.get("pretrain_enabled", False))
        self.chk_adv_sde.setChecked(adv.get("neural_sde_enabled", False))
        self.chk_adv_gflownet.setChecked(adv.get("gflownet_enabled", False))

    def _reset_defaults(self):
        """Reload defaults from disk."""
        try:
            cfg = load_config()
            self._apply_config(cfg)
            logger.info("Settings reset to defaults from disk.")
        except Exception as e:
            logger.error("Failed to reset defaults: %s", e)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class EEGCursorGUI(QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Cursor — Brain-Computer Interface")
        self.setMinimumSize(1280, 860)

        self._project_dir = PROJECT_DIR
        self._config = load_config()
        self._worker: Optional[WorkerThread] = None
        self._board = None
        self._streaming = False
        self._live_running = False

        # Synthetic demo data state
        self._demo_phase = 0.0
        self._demo_probs = {c: 0.0 for c in CLASS_NAMES}

        self._setup_ui()
        self._setup_logging()
        self._connect_signals()
        self._refresh_all_status()

        # Signal display update timer (60 fps)
        self._signal_timer = QTimer()
        self._signal_timer.setInterval(16)  # ~60fps
        self._signal_timer.timeout.connect(self._update_demo_signals)
        self._signal_timer.start()

        # Status refresh timer (every 5 seconds)
        self._status_timer = QTimer()
        self._status_timer.setInterval(5000)
        self._status_timer.timeout.connect(self._refresh_all_status)
        self._status_timer.start()

        logger.info("EEG Cursor GUI started — 5 tabs loaded")

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Build the complete UI layout."""
        self.setStyleSheet(DARK_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Header
        header_row = QHBoxLayout()
        title = QLabel("EEG Cursor")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setStyleSheet("color: #0078d4;")
        header_row.addWidget(title)

        subtitle = QLabel("Pure EEG Brain-Computer Interface")
        subtitle.setStyleSheet("color: #888; font-size: 13px; padding-top: 8px;")
        header_row.addWidget(subtitle)
        header_row.addStretch()

        # Quick status in header
        self._header_status = QLabel("")
        self._header_status.setStyleSheet("color: #666; font-size: 11px; padding-top: 8px;")
        header_row.addWidget(self._header_status)
        main_layout.addLayout(header_row)

        # Splitter: tabs on top, log on bottom
        splitter = QSplitter(Qt.Vertical)

        # Tab widget
        self._tabs = QTabWidget()
        self._tab_dashboard = DashboardTab()
        self._tab_data = DataCollectionTab()
        self._tab_training = TrainingTab()
        self._tab_live = LiveControlTab()
        self._tab_settings = SettingsTab(self._config)

        self._tabs.addTab(self._tab_dashboard, "Dashboard")
        self._tabs.addTab(self._tab_data, "Data Collection")
        self._tabs.addTab(self._tab_training, "Training")
        self._tabs.addTab(self._tab_live, "Live Control")
        self._tabs.addTab(self._tab_settings, "Settings")

        splitter.addWidget(self._tabs)

        # Bottom: Log output
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout(log_group)
        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMinimumHeight(100)
        log_layout.addWidget(self._log_output)

        log_btn_row = QHBoxLayout()
        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.setFixedWidth(100)
        btn_clear_log.clicked.connect(lambda: self._log_output.clear())
        log_btn_row.addWidget(btn_clear_log)
        log_btn_row.addStretch()
        log_layout.addLayout(log_btn_row)

        splitter.addWidget(log_group)
        splitter.setSizes([600, 180])
        main_layout.addWidget(splitter)

        # Status bar
        self._statusbar = QStatusBar()
        self._statusbar.setStyleSheet("color: #888; font-size: 11px;")
        self.setStatusBar(self._statusbar)

    def _setup_logging(self):
        """Route logging to the GUI text edit."""
        handler = QTextEditHandler(self._log_output)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)

    def _connect_signals(self):
        """Wire up all button signals."""
        # Data Collection tab
        self._tab_data.btn_calibrate.clicked.connect(self._on_calibrate)
        self._tab_data.btn_erp.clicked.connect(self._on_erp_trainer)
        self._tab_data.btn_jepa.clicked.connect(self._on_jepa_pretrain)

        # Training tab
        self._tab_training.btn_train.clicked.connect(self._on_train)

        # Live Control tab
        self._tab_live.btn_start.clicked.connect(self._on_launch)
        self._tab_live.btn_stop.clicked.connect(self._on_stop)

    # ------------------------------------------------------------------
    # Status refresh
    # ------------------------------------------------------------------

    def _refresh_all_status(self):
        """Refresh all status indicators across tabs."""
        self._tab_dashboard.refresh_status(self._config)
        self._tab_data.refresh_file_info()
        self._tab_training.refresh_model_info()

        models_dir = Path(self._project_dir) / "models"
        data_dir = Path(self._project_dir) / "data" / "raw"
        models = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.pt"))
        recordings = list(data_dir.glob("*.npz"))

        self._statusbar.showMessage(
            f"Recordings: {len(recordings)} | Models: {len(models)} | "
            f"Project: {self._project_dir}"
        )
        self._header_status.setText(
            f"{len(recordings)} recordings, {len(models)} models"
        )

    # ------------------------------------------------------------------
    # Demo signal update (60fps synthetic data for visual feedback)
    # ------------------------------------------------------------------

    def _update_demo_signals(self):
        """Generate synthetic EEG-like data for display when no board is connected."""
        if self._streaming:
            return  # Real data would be fed from the board

        n_channels = 8
        n_samples = 500
        self._demo_phase += 0.05

        # Generate multi-frequency synthetic EEG
        t = np.linspace(0, 2.0, n_samples)
        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            freq_base = 8.0 + ch * 1.5
            alpha = 30 * np.sin(2 * np.pi * freq_base * t + self._demo_phase + ch)
            beta = 15 * np.sin(2 * np.pi * (freq_base + 10) * t + self._demo_phase * 0.7)
            noise = np.random.randn(n_samples) * 8
            drift = 5 * np.sin(2 * np.pi * 0.3 * t + ch * 0.5)
            data[ch] = alpha + beta + noise + drift

        self._tab_dashboard.signal_display.update_data(data)

        # Update demo classification probabilities (smooth random walk)
        for cls in CLASS_NAMES:
            self._demo_probs[cls] += np.random.randn() * 0.02
            self._demo_probs[cls] = max(0.0, min(1.0, self._demo_probs[cls]))
        # Normalise
        total = sum(self._demo_probs.values())
        if total > 0:
            self._demo_probs = {k: v / total for k, v in self._demo_probs.items()}
        self._tab_dashboard.class_bars.update_probabilities(self._demo_probs)

        # Update cursor position demo
        cx = 0.5 + 0.3 * np.sin(self._demo_phase * 0.3)
        cy = 0.5 + 0.3 * np.cos(self._demo_phase * 0.2)
        self._tab_dashboard.cursor_indicator.set_position(cx, cy)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _run_script(self, args: list, on_done=None):
        """Run a Python script in a worker thread."""
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "Another task is already running.")
            return

        python = sys.executable
        cmd = [python] + args

        self._worker = WorkerThread(cmd, self._project_dir)
        self._worker.output.connect(lambda line: self._log_output.append(line))
        self._worker.finished_signal.connect(
            lambda code: self._on_worker_done(code, on_done)
        )
        self._worker.start()

        self._set_all_buttons_enabled(False)
        logger.info("Running: %s", " ".join(args))

    def _on_worker_done(self, code: int, callback=None):
        """Handle worker thread completion."""
        self._set_all_buttons_enabled(True)
        self._live_running = False
        self._tab_live.set_running(False)
        self._tab_data.lbl_rec_status.setText("Idle")

        if code == 0:
            logger.info("Task completed successfully.")
        else:
            logger.warning("Task finished with exit code %d.", code)

        self._refresh_all_status()

        if callback:
            callback(code)

    def _set_all_buttons_enabled(self, enabled: bool):
        """Enable or disable all action buttons across tabs."""
        # Data tab
        self._tab_data.btn_calibrate.setEnabled(enabled)
        self._tab_data.btn_erp.setEnabled(enabled)
        self._tab_data.btn_jepa.setEnabled(enabled)
        # Training tab
        self._tab_training.btn_train.setEnabled(enabled)
        self._tab_training.btn_browse.setEnabled(enabled)
        self._tab_training.btn_use_latest.setEnabled(enabled)
        # Live tab — handled separately
        if enabled:
            self._tab_live.set_running(False)
        # Settings tab
        self._tab_settings.btn_save_config.setEnabled(enabled)
        self._tab_settings.btn_load_config.setEnabled(enabled)

    def _on_calibrate(self):
        """Run collect_training_data.py."""
        self._tab_data.lbl_rec_status.setText("Recording...")
        self._tab_data.lbl_rec_status.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #4ec94e;"
        )
        self._run_script(
            ["scripts/collect_training_data.py", "--verbose"],
            on_done=lambda code: self._tab_data.lbl_rec_status.setText(
                "Complete" if code == 0 else "Failed"
            ),
        )

    def _on_erp_trainer(self):
        """Run erp_trainer.py."""
        self._tab_data.lbl_rec_status.setText("ERP Training...")
        self._tab_data.lbl_rec_status.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #569cd6;"
        )
        self._run_script(
            ["scripts/erp_trainer.py", "--verbose"],
            on_done=lambda code: self._tab_data.lbl_rec_status.setText(
                "Complete" if code == 0 else "Failed"
            ),
        )

    def _on_jepa_pretrain(self):
        """Run JEPA self-supervised pre-training on unlabeled EEG."""
        self._tab_data.lbl_rec_status.setText("JEPA Pre-Training...")
        self._tab_data.lbl_rec_status.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #ce9178;"
        )
        self._run_script(
            ["scripts/jepa_pretrain.py", "--verbose"],
            on_done=lambda code: self._tab_data.lbl_rec_status.setText(
                "Complete" if code == 0 else "Failed"
            ),
        )

    def _on_train(self):
        """Train a model using the selected type and data file."""
        data_path = self._tab_training.get_data_path()
        if not data_path:
            # Try latest
            data_dir = Path(self._project_dir) / "data" / "raw"
            recordings = sorted(data_dir.glob("*.npz"))
            if not recordings:
                QMessageBox.warning(
                    self, "No Data",
                    "No recordings found. Run calibration or select a file."
                )
                return
            data_path = str(recordings[-1])
            self._tab_training.txt_data_path.setText(data_path)

        model_type = self._tab_training.combo_model.currentText()

        # Show progress
        self._tab_training.progress.setVisible(True)
        self._tab_training.txt_cv_results.clear()

        def on_train_done(code):
            self._tab_training.progress.setVisible(False)
            if code == 0:
                self._tab_training.txt_cv_results.append(
                    '<span style="color:#4ec94e">Training completed successfully.</span>'
                )
                self._tab_training.refresh_model_info()
            else:
                self._tab_training.txt_cv_results.append(
                    '<span style="color:#ff4444">Training failed. Check log for details.</span>'
                )

        self._run_script(
            [
                "scripts/train_model.py",
                "--data-path", data_path,
                "--model-type", model_type,
                "--verbose",
            ],
            on_done=on_train_done,
        )

    def _on_launch(self):
        """Launch the live EEG cursor control."""
        models_dir = Path(self._project_dir) / "models"
        models = sorted(
            list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.pt"))
        )
        if not models:
            QMessageBox.warning(
                self, "No Model",
                "No trained model found. Train one first."
            )
            return

        # Write Live Control toggle states to settings.yaml so the
        # launched script reads the user's current choices.
        try:
            import yaml
            cfg = load_config()
            cfg.setdefault("adaptation", {})
            cfg["adaptation"]["enabled"] = self._tab_live.chk_seal.isChecked()
            cfg["adaptation"]["auto_undo"] = self._tab_live.chk_auto_undo.isChecked()
            cfg.setdefault("advanced", {})
            cfg["advanced"]["state_monitor"] = self._tab_live.chk_state_mon.isChecked()
            cfg["advanced"]["adaptive_routing"] = self._tab_live.chk_adaptive_routing.isChecked()

            settings_path = Path(self._project_dir) / "config" / "settings.yaml"
            with open(settings_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            logger.info(
                "Updated settings.yaml — SEAL=%s, StateMonitor=%s, AutoUndo=%s, AdaptiveRouting=%s",
                self._tab_live.chk_seal.isChecked(),
                self._tab_live.chk_state_mon.isChecked(),
                self._tab_live.chk_auto_undo.isChecked(),
                self._tab_live.chk_adaptive_routing.isChecked(),
            )
        except Exception as e:
            logger.error("Failed to update settings before launch: %s", e)
            QMessageBox.critical(
                self, "Config Error",
                f"Could not write toggle states to settings.yaml:\n{e}\n\n"
                "The script will launch with the previous config."
            )

        latest_model = models[-1]
        args = ["scripts/run_eeg_cursor.py", "--model", str(latest_model), "--verbose"]

        self._live_running = True
        self._tab_live.set_running(True)

        self._run_script(args)

    def _on_stop(self):
        """Stop the currently running process."""
        if self._worker:
            self._worker.stop()
            logger.info("Stopping task...")
            self._live_running = False
            self._tab_live.set_running(False)

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Clean up worker thread on close."""
        self._signal_timer.stop()
        self._status_timer.stop()
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("EEG Cursor")

    window = EEGCursorGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
