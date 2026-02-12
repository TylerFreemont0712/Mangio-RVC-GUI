"""Inference tab — single-file voice conversion."""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from workers.backend_worker import InferenceWorker, ModelLoadWorker

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


class InferenceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._model_worker = None
        self._init_ui()
        self._connect_signals()

    # ── UI construction ───────────────────────────────────────────
    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # Model selection
        model_group = QGroupBox("Model")
        mg = QHBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(_scan_weights())
        self.model_combo.setMinimumWidth(220)
        self.model_combo.setToolTip(
            "Select a .pth voice model from the weights/ folder.\n"
            "The model determines whose voice characteristics are applied."
        )
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Re-scan weights/ and logs/ for new models and indexes.")
        self.load_btn = QPushButton("Load Model")
        self.load_btn.setToolTip(
            "Load the selected model into GPU memory.\n"
            "Must be done before conversion. First load may take several seconds."
        )
        mg.addWidget(QLabel("Voice Model:"))
        mg.addWidget(self.model_combo, 1)
        mg.addWidget(self.refresh_btn)
        mg.addWidget(self.load_btn)
        root.addWidget(model_group)

        # Audio input
        audio_group = QGroupBox("Audio Input")
        ag = QHBoxLayout(audio_group)
        self.audio_path = QLineEdit()
        self.audio_path.setPlaceholderText("Path to audio file ...")
        self.audio_browse = QPushButton("Browse")
        ag.addWidget(self.audio_path, 1)
        ag.addWidget(self.audio_browse)
        root.addWidget(audio_group)

        # Parameters – two-column layout
        params_group = QGroupBox("Parameters")
        pg = QVBoxLayout(params_group)

        # Row 1: transpose + f0 method
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Transpose:"))
        self.transpose_spin = QSpinBox()
        self.transpose_spin.setRange(-24, 24)
        self.transpose_spin.setValue(0)
        self.transpose_spin.setToolTip(
            "Shift pitch in semitones. +12 = one octave up.\n"
            "Male-to-female: try +12. Female-to-male: try -12.\n"
            "Does not affect processing speed."
        )
        r1.addWidget(self.transpose_spin)
        r1.addSpacing(16)
        r1.addWidget(QLabel("F0 Method:"))
        self.f0_combo = QComboBox()
        self.f0_combo.addItems(
            ["pm", "harvest", "crepe", "mangio-crepe", "mangio-crepe-tiny", "rmvpe"]
        )
        self.f0_combo.setToolTip(
            "Pitch detection algorithm — affects quality and speed:\n"
            "  pm: Fastest, lower quality. Good for quick tests.\n"
            "  harvest: Slow, smooth pitch. Good for singing.\n"
            "  crepe: GPU-accelerated, high quality. Needs more VRAM.\n"
            "  mangio-crepe: Crepe variant with adjustable hop length.\n"
            "  rmvpe: Best overall quality and speed balance (recommended)."
        )
        r1.addWidget(self.f0_combo)
        r1.addSpacing(16)
        r1.addWidget(QLabel("Crepe Hop:"))
        self.crepe_hop_spin = QSpinBox()
        self.crepe_hop_spin.setRange(64, 512)
        self.crepe_hop_spin.setValue(160)
        self.crepe_hop_spin.setToolTip(
            "Hop length for crepe/mangio-crepe F0 methods.\n"
            "Lower = finer pitch resolution but slower processing.\n"
            "64: Very detailed, slow. 128: Balanced. 512: Fast, less precise.\n"
            "Only used when F0 method is crepe or mangio-crepe."
        )
        r1.addWidget(self.crepe_hop_spin)
        pg.addLayout(r1)

        # Row 2: filter radius + index
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Filter Radius:"))
        self.filter_spin = QSpinBox()
        self.filter_spin.setRange(0, 7)
        self.filter_spin.setValue(3)
        self.filter_spin.setToolTip(
            "Median filter radius applied to the extracted pitch (F0).\n"
            "Higher values smooth out pitch jitter but may lose\n"
            "expressiveness. 0 = off, 3 = balanced, 7 = very smooth."
        )
        r2.addWidget(self.filter_spin)
        r2.addSpacing(16)
        r2.addWidget(QLabel("Index File:"))
        self.index_combo = QComboBox()
        self.index_combo.setEditable(True)
        self.index_combo.addItems(_scan_indexes())
        self.index_combo.setMinimumWidth(200)
        self.index_combo.setToolTip(
            "FAISS index file (.index) from training.\n"
            "Improves voice similarity to the training data.\n"
            "Found in logs/<experiment>/ after training index step."
        )
        r2.addWidget(self.index_combo, 1)
        pg.addLayout(r2)

        # Row 3: sliders
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Index Rate:"))
        self.index_rate = QDoubleSpinBox()
        self.index_rate.setRange(0.0, 1.0)
        self.index_rate.setSingleStep(0.05)
        self.index_rate.setValue(0.78)
        self.index_rate.setToolTip(
            "Blend between AI output and index retrieval.\n"
            "Higher = voice sounds more like the training data.\n"
            "Too high may introduce artifacts. 0.5-0.8 is typical."
        )
        r3.addWidget(self.index_rate)
        r3.addSpacing(16)
        r3.addWidget(QLabel("RMS Mix:"))
        self.rms_mix = QDoubleSpinBox()
        self.rms_mix.setRange(0.0, 1.0)
        self.rms_mix.setSingleStep(0.05)
        self.rms_mix.setValue(1.0)
        self.rms_mix.setToolTip(
            "Volume envelope mixing rate.\n"
            "0 = use the original audio's volume dynamics.\n"
            "1 = use the converted voice's volume (recommended).\n"
            "Lower values preserve original breathing and dynamics."
        )
        r3.addWidget(self.rms_mix)
        r3.addSpacing(16)
        r3.addWidget(QLabel("Protect:"))
        self.protect = QDoubleSpinBox()
        self.protect.setRange(0.0, 0.5)
        self.protect.setSingleStep(0.01)
        self.protect.setValue(0.33)
        self.protect.setToolTip(
            "Protects voiceless consonants (s, t, p, k, etc.) from artifacts.\n"
            "Lower = stronger protection, keeps consonants crisp.\n"
            "0.33 is a good default. 0.5 = no protection."
        )
        r3.addWidget(self.protect)
        r3.addSpacing(16)
        r3.addWidget(QLabel("Resample:"))
        self.resample_spin = QSpinBox()
        self.resample_spin.setRange(0, 48000)
        self.resample_spin.setValue(0)
        self.resample_spin.setToolTip(
            "Resample the output audio to this sample rate.\n"
            "0 = keep the model's native sample rate (recommended).\n"
            "Common values: 16000, 22050, 44100, 48000."
        )
        r3.addWidget(self.resample_spin)
        pg.addLayout(r3)

        # F0 file
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("F0 Curve File (optional):"))
        self.f0_path = QLineEdit()
        self.f0_path.setPlaceholderText("Leave blank for default F0")
        self.f0_path.setToolTip(
            "Optional custom pitch curve file.\n"
            "Leave blank to auto-extract pitch from the input audio.\n"
            "Advanced: supply a pre-edited F0 curve for manual pitch control."
        )
        self.f0_browse = QPushButton("Browse")
        r4.addWidget(self.f0_path, 1)
        r4.addWidget(self.f0_browse)
        pg.addLayout(r4)

        root.addWidget(params_group)

        # Output
        out_group = QGroupBox("Output")
        og = QVBoxLayout(out_group)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Save to:"))
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("audio-outputs/output.wav")
        self.output_path.setText("audio-outputs/output.wav")
        self.output_browse = QPushButton("Browse")
        out_row.addWidget(self.output_path, 1)
        out_row.addWidget(self.output_browse)
        og.addLayout(out_row)
        root.addWidget(out_group)

        # Convert button
        btn_row = QHBoxLayout()
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.setMinimumHeight(38)
        btn_row.addStretch()
        btn_row.addWidget(self.convert_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Log
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(140)
        root.addWidget(self.log_area)

        root.addStretch()

    # ── Signals / Slots ───────────────────────────────────────────
    def _connect_signals(self):
        self.refresh_btn.clicked.connect(self._refresh_lists)
        self.load_btn.clicked.connect(self._load_model)
        self.audio_browse.clicked.connect(self._browse_audio)
        self.f0_browse.clicked.connect(self._browse_f0)
        self.output_browse.clicked.connect(self._browse_output)
        self.convert_btn.clicked.connect(self._run_inference)

    def _refresh_lists(self):
        self.model_combo.clear()
        self.model_combo.addItems(_scan_weights())
        self.index_combo.clear()
        self.index_combo.addItems(_scan_indexes())
        self.log_area.append("Lists refreshed.")

    def _browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio", "audios", "Audio (*.wav *.mp3 *.flac *.ogg)"
        )
        if path:
            self.audio_path.setText(path)

    def _browse_f0(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select F0 Curve", "")
        if path:
            self.f0_path.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output", "audio-outputs/output.wav", "WAV (*.wav)"
        )
        if path:
            self.output_path.setText(path)

    def _load_model(self):
        name = self.model_combo.currentText()
        if not name:
            self.log_area.append("No model selected.")
            return
        self.log_area.append(f"Loading model: {name} ...")
        self.load_btn.setEnabled(False)
        self._model_worker = ModelLoadWorker(name, self.protect.value(), self.protect.value())
        self._model_worker.result.connect(self._on_model_loaded)
        self._model_worker.error.connect(self._on_error)
        self._model_worker.start()

    def _on_model_loaded(self, res):
        self.load_btn.setEnabled(True)
        self.log_area.append("Model loaded successfully.")

    def _run_inference(self):
        audio = self.audio_path.text().strip()
        if not audio:
            self.log_area.append("Please select an audio file.")
            return
        model = self.model_combo.currentText()
        if not model:
            self.log_area.append("Please load a model first.")
            return
        self.convert_btn.setEnabled(False)
        self.log_area.append("Running inference ...")
        params = {
            "sid": 0,
            "input_audio_path0": audio,
            "input_audio_path1": audio,
            "f0_up_key": self.transpose_spin.value(),
            "f0_file": self.f0_path.text().strip() or None,
            "f0_method": self.f0_combo.currentText(),
            "file_index": "",
            "file_index2": self.index_combo.currentText(),
            "index_rate": self.index_rate.value(),
            "filter_radius": self.filter_spin.value(),
            "resample_sr": self.resample_spin.value(),
            "rms_mix_rate": self.rms_mix.value(),
            "protect": self.protect.value(),
            "crepe_hop_length": self.crepe_hop_spin.value(),
        }
        self._worker = InferenceWorker(params)
        self._worker.result.connect(self._on_inference_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_inference_done(self, info, audio_tuple):
        self.convert_btn.setEnabled(True)
        self.log_area.append(info)
        if audio_tuple and audio_tuple[0] is not None:
            out = self.output_path.text().strip() or "audio-outputs/output.wav"
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            try:
                import soundfile as sf
                sf.write(out, audio_tuple[1], audio_tuple[0])
                self.log_area.append(f"Saved to {out}")
            except Exception as exc:
                self.log_area.append(f"Failed to save: {exc}")

    def _on_error(self, tb):
        self.convert_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.log_area.append(f"ERROR:\n{tb}")
