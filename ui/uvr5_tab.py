"""UVR5 tab — vocal / instrument separation."""

import os

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from workers.backend_worker import BackendWorker, lazy_backend_call

_UVR5_DIR = os.path.join(os.getcwd(), "uvr5_weights")


def _scan_uvr5():
    if not os.path.isdir(_UVR5_DIR):
        return []
    out = []
    for f in sorted(os.listdir(_UVR5_DIR)):
        if f.endswith(".pth") or "onnx" in f:
            out.append(f.replace(".pth", ""))
    return out


class UVR5Tab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # Model
        model_group = QGroupBox("Separation Model")
        mg = QHBoxLayout(model_group)
        mg.addWidget(QLabel("UVR5 Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(_scan_uvr5())
        self.model_combo.setToolTip(
            "UVR5 separation model.\n"
            "HP2: General vocal/instrumental separation.\n"
            "HP5: Main vocal extraction (isolates lead vocals)."
        )
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Re-scan uvr5_weights/ for available models.")
        mg.addWidget(self.model_combo, 1)
        mg.addWidget(self.refresh_btn)
        root.addWidget(model_group)

        # Directories
        dir_group = QGroupBox("Paths")
        dg = QVBoxLayout(dir_group)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Input Directory:"))
        self.input_dir = QLineEdit()
        self.input_dir.setPlaceholderText("Path to audio files")
        self.input_dir.setToolTip("Folder containing audio files to separate.")
        self.input_browse = QPushButton("Browse")
        r1.addWidget(self.input_dir, 1)
        r1.addWidget(self.input_browse)
        dg.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Vocal Output:"))
        self.vocal_dir = QLineEdit("opt/vocal")
        self.vocal_dir.setToolTip("Output folder for extracted vocal tracks.")
        self.vocal_browse = QPushButton("Browse")
        r2.addWidget(self.vocal_dir, 1)
        r2.addWidget(self.vocal_browse)
        dg.addLayout(r2)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Instrument Output:"))
        self.inst_dir = QLineEdit("opt/ins")
        self.inst_dir.setToolTip("Output folder for extracted instrumental tracks.")
        self.inst_browse = QPushButton("Browse")
        r3.addWidget(self.inst_dir, 1)
        r3.addWidget(self.inst_browse)
        dg.addLayout(r3)
        root.addWidget(dir_group)

        # Params
        param_group = QGroupBox("Parameters")
        ppg = QHBoxLayout(param_group)
        ppg.addWidget(QLabel("Aggressiveness:"))
        self.agg_spin = QSpinBox()
        self.agg_spin.setRange(0, 20)
        self.agg_spin.setValue(10)
        self.agg_spin.setToolTip(
            "Separation aggressiveness.\n"
            "Higher = more aggressive vocal/instrument split.\n"
            "Low (5): Gentle, preserves quality but may leave bleed.\n"
            "Medium (10): Balanced (recommended).\n"
            "High (15-20): Cleaner split but may introduce artifacts."
        )
        ppg.addWidget(self.agg_spin)
        ppg.addSpacing(20)
        ppg.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["wav", "flac", "mp3"])
        self.format_combo.setToolTip(
            "Output audio format. wav = lossless (best for further processing).\n"
            "flac = lossless compressed. mp3 = lossy, smallest file size."
        )
        ppg.addWidget(self.format_combo)
        ppg.addStretch()
        root.addWidget(param_group)

        # Run
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Separate")
        self.run_btn.setMinimumHeight(38)
        self.run_btn.setToolTip("Start vocal/instrumental separation on all files in the input directory.")
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Log
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        root.addWidget(self.log_area)

    def _connect_signals(self):
        self.refresh_btn.clicked.connect(self._refresh)
        self.input_browse.clicked.connect(lambda: self._dir_browse(self.input_dir))
        self.vocal_browse.clicked.connect(lambda: self._dir_browse(self.vocal_dir))
        self.inst_browse.clicked.connect(lambda: self._dir_browse(self.inst_dir))
        self.run_btn.clicked.connect(self._run)

    def _refresh(self):
        self.model_combo.clear()
        self.model_combo.addItems(_scan_uvr5())

    def _dir_browse(self, target):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            target.setText(d)

    def _run(self):
        self.run_btn.setEnabled(False)
        self.log_area.clear()
        self._worker = BackendWorker(
            lazy_backend_call,
            args=(
                "uvr",
                self.model_combo.currentText(),
                self.input_dir.text().strip(),
                self.vocal_dir.text().strip(),
                None,
                self.inst_dir.text().strip(),
                self.agg_spin.value(),
                self.format_combo.currentText(),
            ),
            is_generator=True,
        )
        self._worker.log.connect(lambda m: self.log_area.setPlainText(m))
        self._worker.result.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, _):
        self.run_btn.setEnabled(True)
        self.log_area.append("\nSeparation complete.")

    def _on_error(self, tb):
        self.run_btn.setEnabled(True)
        self.log_area.append(f"\nERROR:\n{tb}")
