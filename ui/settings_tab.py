"""Settings tab — formant shifting, device info, configuration."""

import os
import sys

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _scan_formant_presets():
    d = os.path.join(os.getcwd(), "formantshiftcfg")
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f).replace("\\", "/")
        for f in os.listdir(d)
        if f.endswith(".txt")
    )


class SettingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Formant Shifting ──────────────────────────────────────
        fmt_group = QGroupBox("Formant Shifting")
        fg = QVBoxLayout(fmt_group)
        r1 = QHBoxLayout()
        self.formant_enabled = QCheckBox("Enable Formant Shifting")
        r1.addWidget(self.formant_enabled)
        r1.addStretch()
        fg.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Quefrency:"))
        self.quefrency = QDoubleSpinBox()
        self.quefrency.setRange(0.0, 16.0)
        self.quefrency.setSingleStep(0.1)
        self.quefrency.setValue(1.0)
        r2.addWidget(self.quefrency)
        r2.addSpacing(16)
        r2.addWidget(QLabel("Timbre:"))
        self.timbre = QDoubleSpinBox()
        self.timbre.setRange(0.0, 16.0)
        self.timbre.setSingleStep(0.1)
        self.timbre.setValue(1.0)
        r2.addWidget(self.timbre)
        r2.addSpacing(16)
        self.apply_formant_btn = QPushButton("Apply")
        r2.addWidget(self.apply_formant_btn)
        r2.addStretch()
        fg.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        self.preset_combo.addItems(_scan_formant_presets())
        self.preset_refresh = QPushButton("Refresh")
        self.preset_apply_btn = QPushButton("Load Preset")
        r3.addWidget(self.preset_combo, 1)
        r3.addWidget(self.preset_refresh)
        r3.addWidget(self.preset_apply_btn)
        fg.addLayout(r3)
        root.addWidget(fmt_group)

        # ── Device Info ───────────────────────────────────────────
        dev_group = QGroupBox("Device Information")
        devl = QVBoxLayout(dev_group)
        self.device_info = QTextEdit()
        self.device_info.setReadOnly(True)
        self.device_info.setMaximumHeight(180)
        devl.addWidget(self.device_info)
        self._populate_device_info()
        root.addWidget(dev_group)

        # ── Paths ─────────────────────────────────────────────────
        path_group = QGroupBox("Paths")
        pl = QVBoxLayout(path_group)
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Python Cmd:"))
        self.python_cmd = QLineEdit(sys.executable)
        self.python_cmd.setReadOnly(True)
        r4.addWidget(self.python_cmd, 1)
        pl.addLayout(r4)
        r5 = QHBoxLayout()
        r5.addWidget(QLabel("Working Dir:"))
        self.cwd_label = QLineEdit(os.getcwd())
        self.cwd_label.setReadOnly(True)
        r5.addWidget(self.cwd_label, 1)
        pl.addLayout(r5)
        root.addWidget(path_group)

        root.addStretch()

    def _connect_signals(self):
        self.apply_formant_btn.clicked.connect(self._apply_formant)
        self.preset_refresh.clicked.connect(self._refresh_presets)
        self.preset_apply_btn.clicked.connect(self._load_preset)

    def _populate_device_info(self):
        lines = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    lines.append(f"GPU {i}: {name}  ({mem:.1f} GB)")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                lines.append("Device: Apple MPS")
            else:
                lines.append("Device: CPU (no CUDA detected)")
        except Exception as exc:
            lines.append(f"Could not detect GPU: {exc}")
        self.device_info.setPlainText("\n".join(lines))

    def _apply_formant(self):
        try:
            from my_utils import CSVutil
            enabled = self.formant_enabled.isChecked()
            q = self.quefrency.value()
            t = self.timbre.value()
            CSVutil("csvdb/formanting.csv", "w+", "formanting", enabled, q, t)
        except Exception as exc:
            self.device_info.append(f"\nFormant apply error: {exc}")

    def _refresh_presets(self):
        self.preset_combo.clear()
        self.preset_combo.addItems(_scan_formant_presets())

    def _load_preset(self):
        preset = self.preset_combo.currentText().strip()
        if not preset or not os.path.isfile(preset):
            return
        try:
            with open(preset, "r") as f:
                content = f.readlines()
            q = float(content[0].strip())
            t = float(content[1].strip())
            self.quefrency.setValue(q)
            self.timbre.setValue(t)
            self._apply_formant()
        except Exception as exc:
            self.device_info.append(f"\nPreset load error: {exc}")
