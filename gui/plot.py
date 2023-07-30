from __future__ import annotations

from gui.canvas import Canvas
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QScrollArea,
    QWidget
)


class Plot(QWidget):
    def __init__(self):
        super().__init__()

        self.canvas = Canvas()

        self.scroll = QScrollArea(self)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.canvas)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.scroll)
        self.setLayout(self.layout)

        self.setFixedHeight(300)
