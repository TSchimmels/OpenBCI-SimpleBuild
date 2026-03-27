#!/usr/bin/env python3
"""EEG Cursor — Graphical User Interface.

PyQt5 GUI that provides:
  - Dashboard with real-time EEG signal display
  - One-click access to calibration, training, and launch
  - Status indicators for board connection, model
  - Log output window
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
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
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
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
# Log handler that emits to the GUI
# ---------------------------------------------------------------------------
class QTextEditHandler(logging.Handler):
    """Logging handler that writes to a QTextEdit widget."""

    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self._text_edit = text_edit

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        color = {
            "DEBUG": "#888888",
            "INFO": "#CCCCCC",
            "WARNING": "#FFAA00",
            "ERROR": "#FF4444",
            "CRITICAL": "#FF0000",
        }.get(record.levelname, "#CCCCCC")
        self._text_edit.append(f'<span style="color:{color}">{msg}</span>')


# ---------------------------------------------------------------------------
# Worker thread for long-running tasks
# ---------------------------------------------------------------------------
class WorkerThread(QThread):
    """Run a subprocess command and emit output."""

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
# EEG Signal Display
# ---------------------------------------------------------------------------
class SignalDisplay(QWidget):
    """Real-time EEG signal display using pyqtgraph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

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

    def init_channels(self, n_channels: int, channel_names: list = None):
        if not HAS_PYQTGRAPH:
            return
        self._plot_widget.clear()
        self._plots = []
        self._curves = []
        for i in range(min(n_channels, 8)):  # Show max 8 channels
            name = channel_names[i] if channel_names and i < len(channel_names) else f"Ch {i}"
            p = self._plot_widget.addPlot(row=i, col=0)
            p.setLabel("left", name)
            p.showAxis("bottom", i == min(n_channels, 8) - 1)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=False)
            p.setYRange(-100, 100)
            curve = p.plot(pen=pg.mkPen(color=(100, 200, 255), width=1))
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
# Main Window
# ---------------------------------------------------------------------------
class EEGCursorGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Cursor — Brain-Computer Interface")
        self.setMinimumSize(1200, 800)

        self._project_dir = str(Path(__file__).parent.parent)
        self._config = load_config()
        self._worker: Optional[WorkerThread] = None
        self._board = None
        self._streaming = False

        self._setup_ui()
        self._setup_logging()
        self._update_status()

        logger.info("EEG Cursor GUI started")

    def _setup_ui(self):
        """Build the UI layout."""
        # Dark theme
        self.setStyleSheet("""
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
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ccc;
            }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab {
                background-color: #2d2d2d;
                border: 1px solid #444;
                padding: 8px 16px;
                color: #aaa;
            }
            QTabBar::tab:selected { background-color: #1e1e1e; color: #fff; border-bottom: none; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Header
        header = QLabel("EEG Cursor")
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setStyleSheet("color: #0078d4; margin-bottom: 8px;")
        main_layout.addWidget(header)

        subtitle = QLabel("Pure EEG Brain-Computer Interface")
        subtitle.setStyleSheet("color: #888; font-size: 13px; margin-bottom: 12px;")
        main_layout.addWidget(subtitle)

        # Splitter: top (controls + signals) / bottom (log)
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # Top section
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Left: Control panel
        left_panel = self._build_control_panel()
        top_layout.addWidget(left_panel, stretch=1)

        # Right: Signal display
        self._signal_display = SignalDisplay()
        top_layout.addWidget(self._signal_display, stretch=2)

        splitter.addWidget(top)

        # Bottom: Log output
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout(log_group)
        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMaximumHeight(200)
        log_layout.addWidget(self._log_output)
        splitter.addWidget(log_group)

        splitter.setSizes([500, 200])

        # Status bar
        self._statusbar = QStatusBar()
        self._statusbar.setStyleSheet("color: #888; font-size: 12px;")
        self.setStatusBar(self._statusbar)

    def _build_control_panel(self) -> QWidget:
        """Build the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 12, 0)

        # Status indicators
        status_group = QGroupBox("System Status")
        status_layout = QGridLayout(status_group)

        self._lbl_board = QLabel("Disconnected")
        self._lbl_board.setObjectName("status-bad")
        self._lbl_model = QLabel("No model")
        self._lbl_model.setObjectName("status-bad")
        self._lbl_gpu = QLabel("Checking...")

        status_layout.addWidget(QLabel("Board:"), 0, 0)
        status_layout.addWidget(self._lbl_board, 0, 1)
        status_layout.addWidget(QLabel("Model:"), 1, 0)
        status_layout.addWidget(self._lbl_model, 1, 1)
        status_layout.addWidget(QLabel("GPU:"), 2, 0)
        status_layout.addWidget(self._lbl_gpu, 2, 1)

        layout.addWidget(status_group)

        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        self._btn_test = QPushButton("Test Synthetic Pipeline")
        self._btn_test.clicked.connect(self._on_test_synthetic)
        actions_layout.addWidget(self._btn_test)

        self._btn_calibrate = QPushButton("Collect Training Data")
        self._btn_calibrate.clicked.connect(self._on_calibrate)
        actions_layout.addWidget(self._btn_calibrate)

        # Train row
        train_row = QHBoxLayout()
        self._btn_train = QPushButton("Train Model")
        self._btn_train.clicked.connect(self._on_train)
        self._combo_model = QComboBox()
        self._combo_model.addItems(["csp_lda", "riemannian", "eegnet"])
        train_row.addWidget(self._btn_train, stretch=2)
        train_row.addWidget(self._combo_model, stretch=1)
        actions_layout.addLayout(train_row)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #444;")
        actions_layout.addWidget(line)

        # Launch button
        self._btn_launch = QPushButton("LAUNCH EEG CURSOR")
        self._btn_launch.setObjectName("primary")
        self._btn_launch.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self._btn_launch.setMinimumHeight(50)
        self._btn_launch.clicked.connect(self._on_launch)
        actions_layout.addWidget(self._btn_launch)

        self._btn_stop = QPushButton("STOP")
        self._btn_stop.setObjectName("danger")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        actions_layout.addWidget(self._btn_stop)

        layout.addWidget(actions_group)

        # Run Tests
        test_group = QGroupBox("Testing")
        test_layout = QVBoxLayout(test_group)
        self._btn_pytest = QPushButton("Run Unit Tests")
        self._btn_pytest.clicked.connect(self._on_pytest)
        test_layout.addWidget(self._btn_pytest)
        layout.addWidget(test_group)

        layout.addStretch()
        return panel

    def _setup_logging(self):
        """Route logging to the GUI text edit."""
        handler = QTextEditHandler(self._log_output)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)

    def _update_status(self):
        """Refresh status indicators."""
        models_dir = Path(self._project_dir) / "models"
        data_dir = Path(self._project_dir) / "data" / "raw"

        # Model
        models = sorted(models_dir.glob("*.pkl"))
        if models:
            self._lbl_model.setText(models[-1].name)
            self._lbl_model.setObjectName("status-good")
        else:
            self._lbl_model.setText("No model")
            self._lbl_model.setObjectName("status-bad")

        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                self._lbl_gpu.setText(torch.cuda.get_device_name(0))
                self._lbl_gpu.setObjectName("status-good")
            else:
                self._lbl_gpu.setText("CPU only")
                self._lbl_gpu.setObjectName("status-warn")
        except ImportError:
            self._lbl_gpu.setText("PyTorch not installed")
            self._lbl_gpu.setObjectName("status-bad")

        # Recordings count
        n_recordings = len(list(data_dir.glob("*.npz")))
        self._statusbar.showMessage(
            f"Recordings: {n_recordings} | Models: {len(models)} | "
            f"Project: {self._project_dir}"
        )

        # Re-apply styles after objectName changes
        for lbl in [self._lbl_board, self._lbl_model, self._lbl_gpu]:
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

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
        self._worker.finished_signal.connect(lambda code: self._on_worker_done(code, on_done))
        self._worker.start()

        self._set_buttons_enabled(False)
        logger.info("Running: %s", " ".join(args))

    def _on_worker_done(self, code: int, callback=None):
        self._set_buttons_enabled(True)
        if code == 0:
            logger.info("Task completed successfully.")
        else:
            logger.warning("Task finished with exit code %d.", code)
        self._update_status()
        if callback:
            callback(code)

    def _set_buttons_enabled(self, enabled: bool):
        for btn in [self._btn_test, self._btn_calibrate, self._btn_train,
                     self._btn_launch, self._btn_pytest]:
            btn.setEnabled(enabled)
        self._btn_stop.setEnabled(not enabled)

    def _on_test_synthetic(self):
        self._run_script(["scripts/test_synthetic.py", "--verbose"])

    def _on_calibrate(self):
        self._run_script(["scripts/collect_training_data.py", "--verbose"])

    def _on_train(self):
        data_dir = Path(self._project_dir) / "data" / "raw"
        recordings = sorted(data_dir.glob("*.npz"))

        if not recordings:
            QMessageBox.warning(self, "No Data", "No recordings found. Run calibration first.")
            return

        latest = recordings[-1]
        model_type = self._combo_model.currentText()

        self._run_script([
            "scripts/train_model.py",
            "--data-path", str(latest),
            "--model-type", model_type,
            "--verbose",
        ])

    def _on_launch(self):
        models_dir = Path(self._project_dir) / "models"
        models = sorted(models_dir.glob("*.pkl"))

        if not models:
            QMessageBox.warning(self, "No Model", "No trained model found. Train one first.")
            return

        latest_model = models[-1]
        args = ["scripts/run_eeg_cursor.py", "--model", str(latest_model), "--verbose"]

        self._run_script(args)
        self._btn_stop.setEnabled(True)

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
            logger.info("Stopping task...")

    def _on_pytest(self):
        self._run_script(["-m", "pytest", "tests/", "-v", "--tb=short"])

    def closeEvent(self, event):
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
