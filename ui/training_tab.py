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

from workers.backend_worker import BackendWorker, lazy_backend_call


class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._init_ui()
        self._connect_signals()
        self._auto_pretrained()

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
        self.exp_name.setToolTip(
            "Unique name for this training run.\n"
            "Creates a folder under logs/ to store checkpoints and indexes."
        )
        r1.addWidget(self.exp_name, 1)
        r1.addSpacing(12)
        r1.addWidget(QLabel("Version:"))
        self.version_combo = QComboBox()
        self.version_combo.addItems(["v2", "v1"])
        self.version_combo.setToolTip(
            "Model architecture version.\n"
            "v2: Better quality, requires v2 pretrained models.\n"
            "v1: Original architecture, wider pretrained model availability."
        )
        r1.addWidget(self.version_combo)
        eg.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Dataset Directory:"))
        self.dataset_dir = QLineEdit()
        self.dataset_dir.setPlaceholderText("Path to dataset folder")
        self.dataset_dir.setToolTip(
            "Folder with clean audio files for training.\n"
            "Best results: 10-30 min of high-quality speech/singing,\n"
            "no background noise, single speaker, WAV format."
        )
        self.dataset_browse = QPushButton("Browse")
        r2.addWidget(self.dataset_dir, 1)
        r2.addWidget(self.dataset_browse)
        eg.addLayout(r2)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Sample Rate:"))
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(["40k", "48k", "32k"])
        self.sr_combo.setToolTip(
            "Target sample rate. Must match your pretrained models.\n"
            "40k: Most common, good balance. 48k: Highest quality.\n"
            "32k: Lower quality but faster training and inference."
        )
        r3.addWidget(self.sr_combo)
        r3.addSpacing(12)
        r3.addWidget(QLabel("CPU Threads:"))
        self.cpu_spin = QSpinBox()
        self.cpu_spin.setRange(1, cpu_count())
        self.cpu_spin.setValue(min(cpu_count(), 8))
        self.cpu_spin.setToolTip(
            "CPU threads for preprocessing and feature extraction.\n"
            "More threads = faster preprocessing but higher RAM usage.\n"
            "8 is usually a good default for most systems."
        )
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
        self.f0_combo.setToolTip(
            "Pitch extraction method for feature extraction.\n"
            "rmvpe: Best quality and reliability (recommended).\n"
            "crepe: High quality, GPU-accelerated.\n"
            "harvest: Good for singing. pm: Fastest but lowest quality."
        )
        r4.addWidget(self.f0_combo)
        r4.addSpacing(12)
        self.pitch_guidance = QCheckBox("Pitch Guidance (F0)")
        self.pitch_guidance.setChecked(True)
        self.pitch_guidance.setToolTip(
            "Enable pitch (F0) guidance during training.\n"
            "ON: Better for singing and tonal accuracy (recommended).\n"
            "OFF: Slightly faster inference, suitable for speech-only models."
        )
        r4.addWidget(self.pitch_guidance)
        r4.addSpacing(12)
        r4.addWidget(QLabel("Crepe Hop:"))
        self.crepe_hop = QSpinBox()
        self.crepe_hop.setRange(64, 512)
        self.crepe_hop.setValue(128)
        self.crepe_hop.setToolTip(
            "Crepe hop length for pitch extraction.\n"
            "Lower = more detailed F0, slower extraction.\n"
            "128 is a good default for training quality."
        )
        r4.addWidget(self.crepe_hop)
        r4.addSpacing(12)
        r4.addWidget(QLabel("GPU(s):"))
        self.gpu_entry = QLineEdit("0")
        self.gpu_entry.setMaximumWidth(80)
        self.gpu_entry.setToolTip(
            "GPU device index(es) for training.\n"
            "0 = first GPU. Use comma-separated for multi-GPU (e.g. 0,1).\n"
            "Multi-GPU can significantly speed up training."
        )
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
        self.spk_id.setToolTip("Speaker ID for multi-speaker models. Use 0 for single-speaker training.")
        r5.addWidget(self.spk_id)
        r5.addSpacing(12)
        r5.addWidget(QLabel("Batch Size:"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(8)
        self.batch_size.setToolTip(
            "Training batch size. Higher = faster training per epoch.\n"
            "8 is good for 8 GB VRAM. 4 for 4-6 GB. 12+ for 12+ GB.\n"
            "Too high causes out-of-memory errors."
        )
        r5.addWidget(self.batch_size)
        r5.addSpacing(12)
        r5.addWidget(QLabel("Total Epochs:"))
        self.total_epoch = QSpinBox()
        self.total_epoch.setRange(1, 100000)
        self.total_epoch.setValue(200)
        self.total_epoch.setToolTip(
            "Total training epochs. More epochs = better voice quality\n"
            "up to a point (diminishing returns). 200-400 is typical.\n"
            "Overtraining (too many epochs) causes robotic artifacts."
        )
        r5.addWidget(self.total_epoch)
        r5.addSpacing(12)
        r5.addWidget(QLabel("Save Every N Epochs:"))
        self.save_epoch = QSpinBox()
        self.save_epoch.setRange(1, 100000)
        self.save_epoch.setValue(50)
        self.save_epoch.setToolTip(
            "Save a checkpoint every N epochs.\n"
            "Lower values let you recover earlier states but use more disk.\n"
            "50 is a good balance for most training runs."
        )
        r5.addWidget(self.save_epoch)
        tg.addLayout(r5)

        r6 = QHBoxLayout()
        self.save_latest = QCheckBox("Save Only Latest")
        self.save_latest.setChecked(True)
        self.save_latest.setToolTip(
            "Keep only the most recent checkpoint to save disk space.\n"
            "Disable to keep all checkpoints (useful to compare epochs)."
        )
        r6.addWidget(self.save_latest)
        self.cache_gpu = QCheckBox("Cache to GPU")
        self.cache_gpu.setToolTip(
            "Load the full dataset into GPU VRAM for faster training.\n"
            "Speeds up training significantly but requires extra VRAM.\n"
            "Only enable if your GPU has enough free memory."
        )
        r6.addWidget(self.cache_gpu)
        self.save_every_weight = QCheckBox("Save Small Model Every Gen")
        self.save_every_weight.setToolTip(
            "Extract a small inference-ready model at each save point.\n"
            "Lets you test intermediate models without manual extraction.\n"
            "Uses extra disk space."
        )
        r6.addWidget(self.save_every_weight)
        r6.addStretch()
        tg.addLayout(r6)

        r7 = QHBoxLayout()
        r7.addWidget(QLabel("Pretrained G:"))
        self.pretrained_g = QLineEdit()
        self.pretrained_g.setPlaceholderText("Auto-detected")
        self.pretrained_g.setToolTip(
            "Pre-trained Generator model path.\n"
            "Auto-detected based on version, sample rate, and pitch guidance.\n"
            "Using pretrained models dramatically improves quality and speed."
        )
        r7.addWidget(self.pretrained_g, 1)
        r7.addSpacing(8)
        r7.addWidget(QLabel("Pretrained D:"))
        self.pretrained_d = QLineEdit()
        self.pretrained_d.setPlaceholderText("Auto-detected")
        self.pretrained_d.setToolTip(
            "Pre-trained Discriminator model path.\n"
            "Must match the Generator version, sample rate, and pitch setting."
        )
        r7.addWidget(self.pretrained_d, 1)
        tg.addLayout(r7)
        root.addWidget(train_group)

        # ── Action Buttons ────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.preprocess_btn = QPushButton("1. Preprocess")
        self.preprocess_btn.setToolTip("Slice and resample audio files for training.")
        self.extract_btn = QPushButton("2. Extract Features")
        self.extract_btn.setToolTip("Extract F0 pitch and speaker embeddings from preprocessed audio.")
        self.train_btn = QPushButton("3. Train Model")
        self.train_btn.setToolTip("Start model training. Monitor progress in the log below and TensorBoard.")
        self.index_btn = QPushButton("4. Train Index")
        self.index_btn.setToolTip("Build FAISS index for voice timbre retrieval during inference.")
        self.oneclick_btn = QPushButton("One-Click Training")
        self.oneclick_btn.setToolTip("Run all four steps (preprocess, extract, train, index) sequentially.")
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
        self._set_busy(True)
        self.log_area.clear()
        self._worker = BackendWorker(
            lazy_backend_call,
            args=(
                "preprocess_dataset",
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
        self._set_busy(True)
        self.log_area.clear()
        self._worker = BackendWorker(
            lazy_backend_call,
            args=(
                "extract_f0_feature",
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
        self._set_busy(True)
        self.log_area.clear()
        self.log_area.append("Training started ... (check console for live output)")
        self._worker = BackendWorker(
            lazy_backend_call,
            args=(
                "click_train",
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
        self._set_busy(True)
        self.log_area.clear()
        self._worker = BackendWorker(
            lazy_backend_call,
            args=(
                "train_index",
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
        self._set_busy(True)
        self.log_area.clear()
        self._worker = BackendWorker(
            lazy_backend_call,
            args=(
                "train1key",
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
