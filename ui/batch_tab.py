"""Batch inference tab — convert a whole directory of audio files."""

import os

from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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

from workers.backend_worker import BackendWorker

_WEIGHTS_DIR = os.path.join(os.getcwd(), "weights")
_LOGS_DIR = os.path.join(os.getcwd(), "logs")


def _scan_weights():
    if not os.path.isdir(_WEIGHTS_DIR):
        return []
    return sorted(f for f in os.listdir(_WEIGHTS_DIR) if f.endswith(".pth"))


def _scan_indexes():
    indexes = []
    if not os.path.isdir(_LOGS_DIR):
        return indexes
    for root, _dirs, files in os.walk(_LOGS_DIR):
        for f in files:
            if f.endswith(".index") and "trained" not in f:
                indexes.append(os.path.join(root, f).replace("\\", "/"))
    return sorted(indexes)


class BatchTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # Model
        model_group = QGroupBox("Model")
        mg = QHBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(_scan_weights())
        self.refresh_btn = QPushButton("Refresh")
        mg.addWidget(QLabel("Voice Model:"))
        mg.addWidget(self.model_combo, 1)
        mg.addWidget(self.refresh_btn)
        root.addWidget(model_group)

        # Directories
        dir_group = QGroupBox("Directories")
        dg = QVBoxLayout(dir_group)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Input Directory:"))
        self.input_dir = QLineEdit()
        self.input_dir_btn = QPushButton("Browse")
        r1.addWidget(self.input_dir, 1)
        r1.addWidget(self.input_dir_btn)
        dg.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Output Directory:"))
        self.output_dir = QLineEdit()
        self.output_dir.setText("audio-outputs")
        self.output_dir_btn = QPushButton("Browse")
        r2.addWidget(self.output_dir, 1)
        r2.addWidget(self.output_dir_btn)
        dg.addLayout(r2)
        root.addWidget(dir_group)

        # Parameters
        params_group = QGroupBox("Parameters")
        pg = QVBoxLayout(params_group)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Transpose:"))
        self.transpose_spin = QSpinBox()
        self.transpose_spin.setRange(-24, 24)
        r3.addWidget(self.transpose_spin)
        r3.addSpacing(12)
        r3.addWidget(QLabel("F0 Method:"))
        self.f0_combo = QComboBox()
        self.f0_combo.addItems(
            ["pm", "harvest", "crepe", "mangio-crepe", "mangio-crepe-tiny", "rmvpe"]
        )
        r3.addWidget(self.f0_combo)
        r3.addSpacing(12)
        r3.addWidget(QLabel("Crepe Hop:"))
        self.crepe_hop = QSpinBox()
        self.crepe_hop.setRange(64, 512)
        self.crepe_hop.setValue(160)
        r3.addWidget(self.crepe_hop)
        pg.addLayout(r3)

        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Index File:"))
        self.index_combo = QComboBox()
        self.index_combo.setEditable(True)
        self.index_combo.addItems(_scan_indexes())
        r4.addWidget(self.index_combo, 1)
        r4.addSpacing(12)
        r4.addWidget(QLabel("Index Rate:"))
        self.index_rate = QDoubleSpinBox()
        self.index_rate.setRange(0.0, 1.0)
        self.index_rate.setSingleStep(0.05)
        self.index_rate.setValue(0.78)
        r4.addWidget(self.index_rate)
        pg.addLayout(r4)

        r5 = QHBoxLayout()
        r5.addWidget(QLabel("Filter Radius:"))
        self.filter_spin = QSpinBox()
        self.filter_spin.setRange(0, 7)
        self.filter_spin.setValue(3)
        r5.addWidget(self.filter_spin)
        r5.addSpacing(12)
        r5.addWidget(QLabel("RMS Mix:"))
        self.rms_mix = QDoubleSpinBox()
        self.rms_mix.setRange(0.0, 1.0)
        self.rms_mix.setSingleStep(0.05)
        self.rms_mix.setValue(1.0)
        r5.addWidget(self.rms_mix)
        r5.addSpacing(12)
        r5.addWidget(QLabel("Protect:"))
        self.protect = QDoubleSpinBox()
        self.protect.setRange(0.0, 0.5)
        self.protect.setSingleStep(0.01)
        self.protect.setValue(0.33)
        r5.addWidget(self.protect)
        r5.addSpacing(12)
        r5.addWidget(QLabel("Resample:"))
        self.resample_spin = QSpinBox()
        self.resample_spin.setRange(0, 48000)
        r5.addWidget(self.resample_spin)
        r5.addSpacing(12)
        r5.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["wav", "flac", "mp3", "ogg", "aac"])
        r5.addWidget(self.format_combo)
        pg.addLayout(r5)
        root.addWidget(params_group)

        # Convert
        btn_row = QHBoxLayout()
        self.convert_btn = QPushButton("Convert All")
        self.convert_btn.setMinimumHeight(38)
        btn_row.addStretch()
        btn_row.addWidget(self.convert_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Log
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        root.addWidget(self.log_area)

    def _connect_signals(self):
        self.refresh_btn.clicked.connect(self._refresh)
        self.input_dir_btn.clicked.connect(self._browse_input)
        self.output_dir_btn.clicked.connect(self._browse_output)
        self.convert_btn.clicked.connect(self._run_batch)

    def _refresh(self):
        self.model_combo.clear()
        self.model_combo.addItems(_scan_weights())
        self.index_combo.clear()
        self.index_combo.addItems(_scan_indexes())
        self.log_area.append("Lists refreshed.")

    def _browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "Input Directory")
        if d:
            self.input_dir.setText(d)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self.output_dir.setText(d)

    def _run_batch(self):
        import importlib
        backend = importlib.import_module("infer-web")

        self.convert_btn.setEnabled(False)
        self.log_area.clear()
        self.log_area.append("Starting batch conversion ...")

        self._worker = BackendWorker(
            backend.vc_multi,
            args=(
                0,
                self.input_dir.text().strip(),
                self.output_dir.text().strip() or "audio-outputs",
                None,
                self.transpose_spin.value(),
                self.f0_combo.currentText(),
                "",
                self.index_combo.currentText(),
                self.index_rate.value(),
                self.filter_spin.value(),
                self.resample_spin.value(),
                self.rms_mix.value(),
                self.protect.value(),
                self.format_combo.currentText(),
                self.crepe_hop.value(),
            ),
            is_generator=True,
        )
        self._worker.log.connect(lambda m: self.log_area.setPlainText(m))
        self._worker.result.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, _):
        self.convert_btn.setEnabled(True)
        self.log_area.append("\nBatch conversion finished.")

    def _on_error(self, tb):
        self.convert_btn.setEnabled(True)
        self.log_area.append(f"\nERROR:\n{tb}")
