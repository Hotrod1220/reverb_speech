from __future__ import annotations

import gc
import matplotlib.pyplot as plt
import pandas as pd
import sounddevice as sd
import soundfile as sf
import torch

from pathlib import Path
from model.model import Model
from model.prediction import Predictor
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QFileDialog,
    QWidget,
)
from gui.dropdown import Dropdown
from gui.explorer import FileExplorer
from gui.plot import Plot


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.resize(1600, 750)
        self.move(100, 100)

        self.setWindowTitle('Room Classification')

        self.icon = QIcon('asset/icon.png')
        self.setWindowIcon(self.icon)

        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QVBoxLayout(self.widget)

        self.playout = QVBoxLayout()
        self.elayout = QVBoxLayout()

        self.current = None
        self.figure = None

        self.model = None
        self.prediction = None
        self.envelope = None
        self.spectrogram = None
        self.result = None

        self.dropdown = Dropdown()
        self.explorer = FileExplorer()
        self.plot = Plot()

        width = int(self.width() / 2)

        self.dropdown.box.setFixedWidth(width)
        self.explorer.list.setMinimumWidth(width)

        self.mapping = {
            0: 'Large Room',
            1: 'Medium Room',
            2: 'Small Room'
        }

        self.load()

        self.prediction = QLabel(
            'Prediction:',
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter
        )

        self.prediction.setObjectName('Prediction')

        self.elayout.addWidget(self.explorer)

        self.layout.addWidget(self.dropdown)
        self.layout.addWidget(self.plot)
        self.layout.addLayout(self.playout)
        self.layout.addWidget(self.prediction)
        self.layout.addLayout(self.elayout)
        self.layout.setStretch(2, 1)

        self.shortcut = QShortcut(
            QKeySequence.fromString('Return'),
            self
        )

        self.dropdown.box.currentIndexChanged.connect(self.update)
        self.shortcut.activated.connect(self.on_click_predict)
        self.explorer.predict.clicked.connect(self.on_click_predict)
        self.explorer.play.clicked.connect(self.on_click_play)
        self.explorer.browse.clicked.connect(self.on_click_load)

    def update(self) -> None:
        if self.result is None:
            return

        self.current = Path(self.current)
        metadata = self.result.get(self.current.name)

        index = self.dropdown.box.currentIndex()

        if index == 0:
            figure = metadata.get('original').get('spectrogram')
        elif index == 1:
            figure = metadata.get('original').get('envelope')
        else:
            figure = metadata.get('original').get('waveform')

        if figure is None:
            QMessageBox.warning(
                self,
                'Warning',
                'The plot could not be generated.'
            )

            return

        self.plot.canvas.display(figure[0])

        self.explorer.list.setFocus()

    def load(self) -> None:
        root = Path.cwd()

        self.model = Model()
        self.model.device = 'cpu'

        path = root.joinpath('model/state/model.pth')

        state = torch.load(path, map_location = self.model.device)
        self.model.load_state_dict(state)

        self.model.eval()

        csv = root.joinpath('model/annotation.csv')
        self.annotation = pd.read_csv(csv)

        self.predictor = Predictor()
        self.predictor.annotation = self.annotation
        self.predictor.mapping = self.mapping
        self.predictor.model = self.model

    def on_click_load(self) -> None:
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.Directory)

        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            path, *_ = dialog.selectedFiles()

            files = [
                file.as_posix()
                for file in Path(path).glob('*.wav')
                if file.exists() and file.is_file()
            ]

            self.explorer.add(files)

    def on_new_prediction(self) -> None:
        if self.result is not None:
            for k in self.result:
                original = self.result[k].get('original')
                transform = self.result[k].get('transform')

                x = original.get('envelope')
                y = original.get('spectrogram')

                plt.clf()
                plt.close('all')

                del x
                del y

                a = transform.get('envelope')
                b = transform.get('spectrogram')

                plt.clf()
                plt.close('all')

                del a
                del b

            self.plot.canvas.cleanup()
            gc.collect()

    def on_click_predict(self) -> None:
        if self.explorer.list.count() == 0:
            QMessageBox.warning(
                self,
                'Warning',
                'Please select a folder to load.'
            )

            return

        self.on_new_prediction()

        current = self.explorer.list.currentItem().text()
        self.current = Path(current)

        self.result = self.predictor.from_path(self.current)
        metadata = self.result.get(self.current.name)

        prediction = metadata.get('prediction').get('label')
        self.prediction.setText(f"Prediction: {prediction}")

        self.update()

    def on_click_play(self) -> None:
        if self.explorer.list.count() == 0:
            QMessageBox.warning(
                self,
                'Warning',
                'Please select a file.'
            )

            return

        current = self.explorer.list.currentItem().text()
        signal, rate = sf.read(current, dtype='float32')

        sd.play(signal, rate)
