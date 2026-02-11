import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget


class RVCMainWindow(QMainWindow):
    """Main application window for Mangio-RVC-Fork."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mangio-RVC-Fork")
        self.setMinimumSize(900, 600)
        self._init_ui()

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        label = QLabel("Mangio-RVC-Fork")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._cleanup()
        event.accept()

    def _cleanup(self) -> None:
        """Release resources and stop background threads before exit."""
        pass
