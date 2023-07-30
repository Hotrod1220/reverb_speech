from __future__ import annotations

import sys
import glob
import pickle
import sounddevice as sd
import soundfile as sf

from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QGridLayout,
    QWidget,
    QPushButton,
    QListWidget
)

class Files:
    def __init__(self):
        self.get_files()
        self.initialize_data()
        self.selected_song = None

    def initialize_data(self) -> None:
        with open('gui/prediction.pkl', 'rb') as handle:
            output = pickle.load(handle)
            output_dict_value = next(iter(output.items()))[0]
            output_dict = output[output_dict_value]

            self.prediction = output_dict['prediction']

            transform = output_dict['transform']
            self.spectrogram = transform['spectrogram']
            self.wave_form = transform['envelope']

    def play_audio(self) -> None:
        if self.selected_song is not None:
            data, freq_samp = sf.read(self.selected_song, dtype='float32')
            sd.play(data, freq_samp)

    def get_files(self) -> None:
        path = Path.cwd()

        path = str(path) + '/sound/*.wav'

        self.files = glob.glob(path)

    def add_files(self, list_files: QListWidget) -> None:
        count = 1
        sound = ""

        for file in self.files:
            i = len(file) - 1
            while i > 0:
                if file[i] == '/':
                    break
                sound = file[i] + sound
                i -= 1

            list_files.insertItem(count, sound)
            count += 1

            if count % 11 == 0:
                print("Maximum of 10 files, the first 10 were selected.")
                break
        return list_files

class MplCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        parent = None,
        width: int = 5,
        height: int = 4,
        dpi: int = 100
    ):
        fig = Figure(
            figsize = (width, height),
            dpi = dpi
        )

        self.axes = fig.add_subplot(111)

        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Room Classification")
        self.geometry()

        self.initialize_style()
        self.file_system = Files()
        self.generate_figures()
        self.generate_button_layout()

        layout = QVBoxLayout()
        layout.addWidget(self.title)

        layout.addWidget(self.wave_form_title)
        layout.addWidget(self.wave_form)

        layout.addWidget(self.spectro_title)
        layout.addWidget(self.spectrogram)

        layout.addLayout(self.button_layout)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def geometry(self) -> None:
        screen = (
            QGuiApplication
            .primaryScreen()
            .availableGeometry()
        )
        self.setGeometry(0, 0, screen.width(), screen.height())

    def initialize_style(self) -> None:
        path = str(Path.cwd())
        path = path + "/gui/styles.css"
        with open(path, "r") as file:
            app.setStyleSheet(file.read())

    def display_prediction(self) -> None:
        self.prediction_title.setText(f"Prediction: {self.prediction['label']}")

    def generate_figures(self) -> None:
        self.prediction = self.file_system.prediction
        spectrogram = self.file_system.spectrogram
        wave_form = self.file_system.wave_form

        self.title = QLabel(
            "Room Classification",
            alignment = Qt.AlignmentFlag.AlignHCenter,
        )

        self.wave_form_title = QLabel(
            "Wave Form:",
            alignment = Qt.AlignmentFlag.AlignTop
        )
        self.wave_form = MplCanvas(wave_form)

        self.spectro_title = QLabel(
            "Spectrogram:",
            alignment = Qt.AlignmentFlag.AlignTop
        )
        self.spectrogram = MplCanvas(spectrogram)

        self.prediction_title = QLabel(
            "Prediction:",
            alignment = Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter
        )

        self.select_file = QLabel(
            "Select File:",
            alignment = Qt.AlignmentFlag.AlignTop
        )

    def generate_button_layout(self) -> None:
        predict_button = QPushButton("Predict Room")
        predict_button.clicked.connect(self.display_prediction)

        play_button = QPushButton("Play")
        play_button.clicked.connect(self.file_system.play_audio)


        self.button_layout = QGridLayout()
        self.button_layout.addWidget(self.prediction_title, 0, 1)
        self.button_layout.addWidget(self.select_file, 0, 2)
        self.button_layout.addWidget(predict_button, 1, 0)
        self.button_layout.addWidget(play_button, 1, 1)

        list_files = QListWidget()
        self.list_files = self.file_system.add_files(list_files)
        self.list_files.clicked.connect(self.get_list_item)
        self.button_layout.addWidget(list_files, 1, 2)

    def get_list_item(self) -> None:
        item = self.list_files.currentItem()
        selected_song_title = "sound/" + item.text()

        path = Path.cwd()

        song_path = path.joinpath(selected_song_title)
        self.file_system.selected_song = song_path

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec()
