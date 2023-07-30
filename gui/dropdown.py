from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QSizePolicy,
    QWidget
)


class Dropdown(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedHeight(50)

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.box = QComboBox()
        self.box.addItem('Spectrogram')
        self.box.addItem('Envelope')
        self.box.addItem('Waveform')

        self.box.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )

        self.layout.addWidget(self.box)
