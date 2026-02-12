"""Main application window — assembles all tabs."""

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from ui.inference_tab import InferenceTab
from ui.batch_tab import BatchTab
from ui.training_tab import TrainingTab
from ui.model_tools_tab import ModelToolsTab
from ui.uvr5_tab import UVR5Tab
from ui.settings_tab import SettingsTab


class RVCMainWindow(QMainWindow):
    """Main application window for Mangio-RVC-Fork."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mangio-RVC-Fork")
        self.setMinimumSize(1060, 700)
        self._workers = []
        self._init_ui()

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        tabs = QTabWidget()

        self.inference_tab = InferenceTab()
        self.batch_tab = BatchTab()
        self.training_tab = TrainingTab()
        self.model_tools_tab = ModelToolsTab()
        self.uvr5_tab = UVR5Tab()
        self.settings_tab = SettingsTab()

        tabs.addTab(self.inference_tab, "Inference")
        tabs.addTab(self.batch_tab, "Batch")
        tabs.addTab(self.training_tab, "Training")
        tabs.addTab(self.model_tools_tab, "Model Tools")
        tabs.addTab(self.uvr5_tab, "UVR5")
        tabs.addTab(self.settings_tab, "Settings")

        layout.addWidget(tabs)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._cleanup()
        event.accept()

    def _cleanup(self) -> None:
        """Release resources and stop background threads before exit."""
        all_tabs = (
            self.inference_tab, self.batch_tab, self.training_tab,
            self.model_tools_tab, self.uvr5_tab, self.settings_tab,
        )
        for tab in all_tabs:
            for attr in ("_worker", "_model_worker"):
                worker = getattr(tab, attr, None)
                if worker is not None and worker.isRunning():
                    if hasattr(worker, "abort"):
                        worker.abort()
                    worker.quit()
                    worker.wait(3000)
