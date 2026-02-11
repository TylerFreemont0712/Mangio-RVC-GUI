"""Training tab — preprocess, extract features, train, build index."""

import os
from multiprocessing import cpu_count

from PyQt6.QtWidgets import (
    QCheckBox,
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

from workers.backend_worker import BackendWorker


class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Experiment ────────────────────────────────────────────
        exp_group = QGroupBox("Experiment")
        eg = QVBoxLayout(exp_group)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Experiment Name:"))
        self.exp_name = QLineEdit()
        self.exp_name.setPlaceholderText("my-voice-model")
        r1.addWidget(self.exp_name, 1)
        r1.addSpacing(12)
        r1.addWidget(QLabel("Version:"))
        self.version_combo = QComboBox()
        self.version_combo.addItems(["v2", "v1"])
        r1.addWidget(self.version_combo)
        eg.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Dataset Directory:"))
        self.dataset_dir = QLineEdit()
        self.dataset_dir.setPlaceholderText("Path to dataset folder")
        self.dataset_browse = QPushButton("Browse")
        r2.addWidget(self.dataset_dir, 1)
        r2.addWidget(self.dataset_browse)
        eg.addLayout(r2)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Sample Rate:"))
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(["40k", "48k", "32k"])
        r3.addWidget(self.sr_combo)
        r3.addSpacing(12)
        r3.addWidget(QLabel("CPU Threads:"))
        self.cpu_spin = QSpinBox()
        self.cpu_spin.setRange(1, cpu_count())
        self.cpu_spin.setValue(min(cpu_count(), 8))
        r3.addWidget(self.cpu_spin)
        r3.addStretch()
        eg.addLayout(r3)
        root.addWidget(exp_group)

        # ── Feature Extraction ────────────────────────────────────
        feat_group = QGroupBox("Feature Extraction")
        fg = QVBoxLayout(feat_group)
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("F0 Method:"))
        self.f0_combo = QComboBox()
        self.f0_combo.addItems(
            ["pm", "harvest", "crepe", "mangio-crepe", "mangio-crepe-tiny", "rmvpe"]
        )
        self.f0_combo.setCurrentText("rmvpe")
        r4.addWidget(self.f0_combo)
        r4.addSpacing(12)
        self.pitch_guidance = QCheckBox("Pitch Guidance (F0)")
        self.pitch_guidance.setChecked(True)
        r4.addWidget(self.pitch_guidance)
        r4.addSpacing(12)
        r4.addWidget(QLabel("Crepe Hop:"))
        self.crepe_hop = QSpinBox()
        self.crepe_hop.setRange(64, 512)
        self.crepe_hop.setValue(128)
        r4.addWidget(self.crepe_hop)
        r4.addSpacing(12)
        r4.addWidget(QLabel("GPU(s):"))
        self.gpu_entry = QLineEdit("0")
        self.gpu_entry.setMaximumWidth(80)
        r4.addWidget(self.gpu_entry)
        r4.addStretch()
        fg.addLayout(r4)
        root.addWidget(feat_group)

        # ── Training ──────────────────────────────────────────────
        train_group = QGroupBox("Training")
        tg = QVBoxLayout(train_group)
        r5 = QHBoxLayout()
        r5.addWidget(QLabel("Speaker ID:"))
        self.spk_id = QSpinBox()
        self.spk_id.setRange(0, 4096)
        r5.addWidget(self.spk_id)
        r5.addSpacing(12)
        r5.addWidget(QLabel("Batch Size:"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(8)
        r5.addWidget(self.batch_size)
        r5.addSpacing(12)
        r5.addWidget(QLabel("Total Epochs:"))
        self.total_epoch = QSpinBox()
        self.total_epoch.setRange(1, 100000)
        self.total_epoch.setValue(200)
        r5.addWidget(self.total_epoch)
        r5.addSpacing(12)
        r5.addWidget(QLabel("Save Every N Epochs:"))
        self.save_epoch = QSpinBox()
        self.save_epoch.setRange(1, 100000)
        self.save_epoch.setValue(50)
        r5.addWidget(self.save_epoch)
        tg.addLayout(r5)

        r6 = QHBoxLayout()
        self.save_latest = QCheckBox("Save Only Latest")
        self.save_latest.setChecked(True)
        r6.addWidget(self.save_latest)
        self.cache_gpu = QCheckBox("Cache to GPU")
        r6.addWidget(self.cache_gpu)
        self.save_every_weight = QCheckBox("Save Small Model Every Gen")
        r6.addWidget(self.save_every_weight)
        r6.addStretch()
        tg.addLayout(r6)

        r7 = QHBoxLayout()
        r7.addWidget(QLabel("Pretrained G:"))
        self.pretrained_g = QLineEdit()
        self.pretrained_g.setPlaceholderText("Auto-detected")
        r7.addWidget(self.pretrained_g, 1)
        r7.addSpacing(8)
        r7.addWidget(QLabel("Pretrained D:"))
        self.pretrained_d = QLineEdit()
        self.pretrained_d.setPlaceholderText("Auto-detected")
        r7.addWidget(self.pretrained_d, 1)
        tg.addLayout(r7)
        root.addWidget(train_group)

        # ── Action Buttons ────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.preprocess_btn = QPushButton("1. Preprocess")
        self.extract_btn = QPushButton("2. Extract Features")
        self.train_btn = QPushButton("3. Train Model")
        self.index_btn = QPushButton("4. Train Index")
        self.oneclick_btn = QPushButton("One-Click Training")
        for b in (self.preprocess_btn, self.extract_btn, self.train_btn, self.index_btn):
            b.setMinimumHeight(34)
            btn_row.addWidget(b)
        root.addLayout(btn_row)
        one_row = QHBoxLayout()
        self.oneclick_btn.setMinimumHeight(38)
        one_row.addStretch()
        one_row.addWidget(self.oneclick_btn)
        one_row.addStretch()
        root.addLayout(one_row)

        # ── Log ───────────────────────────────────────────────────
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        root.addWidget(self.log_area)

    def _connect_signals(self):
        self.dataset_browse.clicked.connect(self._browse_dataset)
        self.preprocess_btn.clicked.connect(self._preprocess)
        self.extract_btn.clicked.connect(self._extract)
        self.train_btn.clicked.connect(self._train)
        self.index_btn.clicked.connect(self._train_index)
        self.oneclick_btn.clicked.connect(self._oneclick)
        self.sr_combo.currentTextChanged.connect(self._auto_pretrained)
        self.version_combo.currentTextChanged.connect(self._auto_pretrained)
        self.pitch_guidance.stateChanged.connect(self._auto_pretrained)

    def _browse_dataset(self):
        d = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if d:
            self.dataset_dir.setText(d)

    def _auto_pretrained(self):
        sr = self.sr_combo.currentText()
        ver = self.version_combo.currentText()
        f0 = "f0" if self.pitch_guidance.isChecked() else ""
        path_str = "" if ver == "v1" else "_v2"
        g = "pretrained%s/%sG%s.pth" % (path_str, f0, sr)
        d = "pretrained%s/%sD%s.pth" % (path_str, f0, sr)
        self.pretrained_g.setText(g if os.path.isfile(g) else "")
        self.pretrained_d.setText(d if os.path.isfile(d) else "")

    def _set_busy(self, busy):
        for b in (self.preprocess_btn, self.extract_btn, self.train_btn,
                  self.index_btn, self.oneclick_btn):
            b.setEnabled(not busy)

    def _preprocess(self):
        import importlib
        backend = importlib.import_module("infer-web")
        self._set_busy(True)
        self.log_area.clear()
        self._worker = BackendWorker(
            backend.preprocess_dataset,
            args=(
                self.dataset_dir.text().strip(),
                self.exp_name.text().strip(),
                self.sr_combo.currentText(),
                self.cpu_spin.value(),
            ),
            is_generator=True,
        )
        self._worker.log.connect(lambda m: self.log_area.setPlainText(m))
        self._worker.result.connect(lambda _: self._set_busy(False))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _extract(self):
        import importlib
        backend = importlib.import_module("infer-web")
        self._set_busy(True)
        self.log_area.clear()
        self._worker = BackendWorker(
            backend.extract_f0_feature,
            args=(
                self.gpu_entry.text().strip(),
                self.cpu_spin.value(),
                self.f0_combo.currentText(),
                self.pitch_guidance.isChecked(),
                self.exp_name.text().strip(),
                self.version_combo.currentText(),
                self.crepe_hop.value(),
            ),
            is_generator=True,
        )
        self._worker.log.connect(lambda m: self.log_area.setPlainText(m))
        self._worker.result.connect(lambda _: self._set_busy(False))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _train(self):
        import importlib
        backend = importlib.import_module("infer-web")
        self._set_busy(True)
        self.log_area.clear()
        self.log_area.append("Training started ... (check console for live output)")
        self._worker = BackendWorker(
            backend.click_train,
            args=(
                self.exp_name.text().strip(),
                self.sr_combo.currentText(),
                self.pitch_guidance.isChecked(),
                self.spk_id.value(),
                self.save_epoch.value(),
                self.total_epoch.value(),
                self.batch_size.value(),
                self.save_latest.isChecked(),
                self.pretrained_g.text().strip(),
                self.pretrained_d.text().strip(),
                self.gpu_entry.text().strip(),
                self.cache_gpu.isChecked(),
                self.save_every_weight.isChecked(),
                self.version_combo.currentText(),
            ),
        )
        self._worker.result.connect(self._on_train_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_train_done(self, res):
        self._set_busy(False)
        if isinstance(res, tuple):
            self.log_area.append(str(res[0]))
        else:
            self.log_area.append(str(res))

    def _train_index(self):
        import importlib
        backend = importlib.import_module("infer-web")
        self._set_busy(True)
        self.log_area.clear()
        self._worker = BackendWorker(
            backend.train_index,
            args=(
                self.exp_name.text().strip(),
                self.version_combo.currentText(),
            ),
            is_generator=True,
        )
        self._worker.log.connect(lambda m: self.log_area.setPlainText(m))
        self._worker.result.connect(lambda _: self._set_busy(False))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _oneclick(self):
        import importlib
        backend = importlib.import_module("infer-web")
        self._set_busy(True)
        self.log_area.clear()
        self._worker = BackendWorker(
            backend.train1key,
            args=(
                self.exp_name.text().strip(),
                self.sr_combo.currentText(),
                self.pitch_guidance.isChecked(),
                self.dataset_dir.text().strip(),
                self.spk_id.value(),
                self.cpu_spin.value(),
                self.f0_combo.currentText(),
                self.save_epoch.value(),
                self.total_epoch.value(),
                self.batch_size.value(),
                self.save_latest.isChecked(),
                self.pretrained_g.text().strip(),
                self.pretrained_d.text().strip(),
                self.gpu_entry.text().strip(),
                self.cache_gpu.isChecked(),
                self.save_every_weight.isChecked(),
                self.version_combo.currentText(),
                self.crepe_hop.value(),
            ),
            is_generator=True,
        )
        self._worker.log.connect(lambda m: self.log_area.setPlainText(m))
        self._worker.result.connect(lambda _: self._set_busy(False))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_error(self, tb):
        self._set_busy(False)
        self.log_area.append(f"\nERROR:\n{tb}")
