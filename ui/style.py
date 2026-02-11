"""Centralised Catppuccin-Mocha-inspired stylesheet for the application."""

STYLESHEET = """
/* ── Base ─────────────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-size: 13px;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 4px;
    background: #1e1e2e;
    top: -1px;
}
QTabBar::tab {
    background: #313244;
    color: #a6adc8;
    padding: 8px 18px;
    border: 1px solid #45475a;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #1e1e2e;
    color: #89b4fa;
    border-bottom: 2px solid #89b4fa;
}
QTabBar::tab:hover:!selected {
    background: #45475a;
}

/* ── Buttons ──────────────────────────────────────────────────────── */
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    padding: 7px 18px;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton:hover { background-color: #74c7ec; }
QPushButton:pressed { background-color: #585b70; color: #cdd6f4; }
QPushButton:disabled { background-color: #45475a; color: #6c7086; }
QPushButton[flat="true"] {
    background-color: transparent;
    color: #89b4fa;
    padding: 4px 8px;
}

/* ── Inputs ───────────────────────────────────────────────────────── */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    padding: 5px 8px;
    border-radius: 4px;
    min-height: 22px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #89b4fa;
}
QComboBox::drop-down { border: none; padding-right: 8px; }
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #45475a;
    border: 1px solid #585b70;
}

/* ── Group boxes ──────────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 4px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: bold;
    color: #a6adc8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

/* ── Sliders ──────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: #45475a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 2px;
}

/* ── Text areas ───────────────────────────────────────────────────── */
QTextEdit, QPlainTextEdit {
    background-color: #181825;
    color: #a6adc8;
    border: 1px solid #45475a;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
}

/* ── Check / Radio ────────────────────────────────────────────────── */
QCheckBox, QRadioButton { color: #cdd6f4; spacing: 6px; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px; height: 16px;
    border: 1px solid #45475a;
    border-radius: 3px;
    background: #313244;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background: #89b4fa;
    border: 1px solid #89b4fa;
}

/* ── Labels ───────────────────────────────────────────────────────── */
QLabel { color: #cdd6f4; }
QLabel[heading="true"] { font-size: 15px; font-weight: bold; color: #89b4fa; }

/* ── Scroll bars ──────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #181825; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #45475a; border-radius: 4px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #181825; height: 8px; border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #45475a; border-radius: 4px; min-width: 20px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""
