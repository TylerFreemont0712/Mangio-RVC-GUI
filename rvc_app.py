"""Mangio-RVC-Fork — single authoritative GUI entry point.

Usage::

    python rvc_app.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# High-DPI awareness — must be set before QApplication is created
# ---------------------------------------------------------------------------
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("Mangio-RVC-Fork")
    app.setStyle("Fusion")

    # Centralised stylesheet (Catppuccin Mocha dark theme)
    from ui.style import STYLESHEET
    app.setStyleSheet(STYLESHEET)

    from ui.main_window import RVCMainWindow

    window = RVCMainWindow()
    window.show()

    exit_code = app.exec()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
