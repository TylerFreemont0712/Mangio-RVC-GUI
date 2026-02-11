"""Model Tools tab — extract, merge, info, export ONNX."""

import os

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from workers.backend_worker import BackendWorker


class ModelToolsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Model Info ────────────────────────────────────────────
        info_group = QGroupBox("Model Information")
        ig = QVBoxLayout(info_group)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Model Path:"))
        self.info_path = QLineEdit()
        self.info_path.setPlaceholderText("weights/model.pth")
        self.info_browse = QPushButton("Browse")
        self.info_btn = QPushButton("Show Info")
        r1.addWidget(self.info_path, 1)
        r1.addWidget(self.info_browse)
        r1.addWidget(self.info_btn)
        ig.addLayout(r1)
        root.addWidget(info_group)

        # ── Extract Small Model ───────────────────────────────────
        ext_group = QGroupBox("Extract Small Model")
        eg = QVBoxLayout(ext_group)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Checkpoint:"))
        self.ext_path = QLineEdit()
        self.ext_path.setPlaceholderText("logs/experiment/G_xxxx.pth")
        self.ext_browse = QPushButton("Browse")
        r2.addWidget(self.ext_path, 1)
        r2.addWidget(self.ext_browse)
        eg.addLayout(r2)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Save Name:"))
        self.ext_name = QLineEdit()
        self.ext_name.setPlaceholderText("MyModel")
        r3.addWidget(self.ext_name, 1)
        r3.addSpacing(12)
        r3.addWidget(QLabel("SR:"))
        self.ext_sr = QComboBox()
        self.ext_sr.addItems(["40k", "48k", "32k"])
        r3.addWidget(self.ext_sr)
        r3.addSpacing(12)
        self.ext_f0 = QCheckBox("Pitch Guidance")
        self.ext_f0.setChecked(True)
        r3.addWidget(self.ext_f0)
        r3.addSpacing(12)
        r3.addWidget(QLabel("Version:"))
        self.ext_ver = QComboBox()
        self.ext_ver.addItems(["v2", "v1"])
        r3.addWidget(self.ext_ver)
        eg.addLayout(r3)
        r3b = QHBoxLayout()
        r3b.addWidget(QLabel("Info:"))
        self.ext_info = QLineEdit()
        self.ext_info.setPlaceholderText("Optional model description")
        r3b.addWidget(self.ext_info, 1)
        r3b.addSpacing(12)
        self.ext_btn = QPushButton("Extract")
        r3b.addWidget(self.ext_btn)
        eg.addLayout(r3b)
        root.addWidget(ext_group)

        # ── Merge Models ──────────────────────────────────────────
        merge_group = QGroupBox("Merge Models")
        mrg = QVBoxLayout(merge_group)
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Model A:"))
        self.merge_a = QLineEdit()
        self.merge_a_browse = QPushButton("Browse")
        r4.addWidget(self.merge_a, 1)
        r4.addWidget(self.merge_a_browse)
        r4.addSpacing(8)
        r4.addWidget(QLabel("Model B:"))
        self.merge_b = QLineEdit()
        self.merge_b_browse = QPushButton("Browse")
        r4.addWidget(self.merge_b, 1)
        r4.addWidget(self.merge_b_browse)
        mrg.addLayout(r4)
        r5 = QHBoxLayout()
        r5.addWidget(QLabel("Alpha (A weight):"))
        self.merge_alpha = QDoubleSpinBox()
        self.merge_alpha.setRange(0.0, 1.0)
        self.merge_alpha.setSingleStep(0.1)
        self.merge_alpha.setValue(0.5)
        r5.addWidget(self.merge_alpha)
        r5.addSpacing(12)
        r5.addWidget(QLabel("SR:"))
        self.merge_sr = QComboBox()
        self.merge_sr.addItems(["40k", "48k", "32k"])
        r5.addWidget(self.merge_sr)
        r5.addSpacing(12)
        self.merge_f0 = QCheckBox("Pitch Guidance")
        self.merge_f0.setChecked(True)
        r5.addWidget(self.merge_f0)
        r5.addSpacing(12)
        r5.addWidget(QLabel("Version:"))
        self.merge_ver = QComboBox()
        self.merge_ver.addItems(["v2", "v1"])
        r5.addWidget(self.merge_ver)
        mrg.addLayout(r5)
        r5b = QHBoxLayout()
        r5b.addWidget(QLabel("Save Name:"))
        self.merge_name = QLineEdit()
        self.merge_name.setPlaceholderText("merged_model")
        r5b.addWidget(self.merge_name, 1)
        r5b.addSpacing(12)
        r5b.addWidget(QLabel("Info:"))
        self.merge_info = QLineEdit()
        r5b.addWidget(self.merge_info, 1)
        r5b.addSpacing(12)
        self.merge_btn = QPushButton("Merge")
        r5b.addWidget(self.merge_btn)
        mrg.addLayout(r5b)
        root.addWidget(merge_group)

        # ── Export ONNX ───────────────────────────────────────────
        onnx_group = QGroupBox("Export ONNX")
        og = QHBoxLayout(onnx_group)
        og.addWidget(QLabel("Model:"))
        self.onnx_src = QLineEdit()
        self.onnx_src.setPlaceholderText("weights/model.pth")
        self.onnx_src_browse = QPushButton("Browse")
        og.addWidget(self.onnx_src, 1)
        og.addWidget(self.onnx_src_browse)
        og.addSpacing(8)
        og.addWidget(QLabel("Output:"))
        self.onnx_dst = QLineEdit()
        self.onnx_dst.setPlaceholderText("model.onnx")
        self.onnx_dst_browse = QPushButton("Browse")
        og.addWidget(self.onnx_dst, 1)
        og.addWidget(self.onnx_dst_browse)
        og.addSpacing(8)
        self.onnx_btn = QPushButton("Export")
        og.addWidget(self.onnx_btn)
        root.addWidget(onnx_group)

        # ── Log ───────────────────────────────────────────────────
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(160)
        root.addWidget(self.log_area)
        root.addStretch()

    def _connect_signals(self):
        self.info_browse.clicked.connect(lambda: self._browse(self.info_path))
        self.info_btn.clicked.connect(self._show_info)
        self.ext_browse.clicked.connect(lambda: self._browse(self.ext_path))
        self.ext_btn.clicked.connect(self._extract_model)
        self.merge_a_browse.clicked.connect(lambda: self._browse(self.merge_a))
        self.merge_b_browse.clicked.connect(lambda: self._browse(self.merge_b))
        self.merge_btn.clicked.connect(self._merge_models)
        self.onnx_src_browse.clicked.connect(lambda: self._browse(self.onnx_src))
        self.onnx_dst_browse.clicked.connect(
            lambda: self._save_browse(self.onnx_dst, "ONNX (*.onnx)")
        )
        self.onnx_btn.clicked.connect(self._export_onnx)

    def _browse(self, target: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch (*.pth)")
        if path:
            target.setText(path)

    def _save_browse(self, target: QLineEdit, filt: str):
        path, _ = QFileDialog.getSaveFileName(self, "Save As", "", filt)
        if path:
            target.setText(path)

    def _show_info(self):
        from train.process_ckpt import show_info
        result = show_info(self.info_path.text().strip())
        self.log_area.setPlainText(result)

    def _extract_model(self):
        from train.process_ckpt import extract_small_model
        self.log_area.clear()
        self.ext_btn.setEnabled(False)
        self._worker = BackendWorker(
            extract_small_model,
            args=(
                self.ext_path.text().strip(),
                self.ext_name.text().strip(),
                self.ext_sr.currentText(),
                1 if self.ext_f0.isChecked() else 0,
                self.ext_info.text().strip(),
                self.ext_ver.currentText(),
            ),
        )
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _merge_models(self):
        from train.process_ckpt import merge
        self.log_area.clear()
        self.merge_btn.setEnabled(False)
        self._worker = BackendWorker(
            merge,
            args=(
                self.merge_a.text().strip(),
                self.merge_b.text().strip(),
                self.merge_alpha.value(),
                self.merge_sr.currentText(),
                self.merge_f0.isChecked(),
                self.merge_info.text().strip(),
                self.merge_name.text().strip(),
                self.merge_ver.currentText(),
            ),
        )
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _export_onnx(self):
        import importlib
        backend = importlib.import_module("infer-web")
        self.log_area.clear()
        self.onnx_btn.setEnabled(False)
        self._worker = BackendWorker(
            backend.export_onnx,
            args=(self.onnx_src.text().strip(), self.onnx_dst.text().strip()),
        )
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, res):
        for b in (self.ext_btn, self.merge_btn, self.onnx_btn):
            b.setEnabled(True)
        self.log_area.append(str(res))

    def _on_error(self, tb):
        for b in (self.ext_btn, self.merge_btn, self.onnx_btn):
            b.setEnabled(True)
        self.log_area.append(f"ERROR:\n{tb}")
