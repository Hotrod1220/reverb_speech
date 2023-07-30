from __future__ import annotations

import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class Canvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure()

        self.ax = self.figure.add_subplot(
            111,
            autoscale_on=True
        )

        super().__init__(self.figure)

        self.figure.patch.set_facecolor('#222222')
        self.ax.patch.set_facecolor('#222222')

        self.ax.set_axis_off()

        self.draw()

    def cleanup(self) -> None:
        self.ax.cla()

        for ax in self.figure.axes:
            ax.cla()

        plt.close(self.figure)

    def display(self, figure: Figure) -> None:
        self.figure = figure
        self.figure.patch.set_facecolor('#222222')

        for ax in self.figure.axes:
            ax.set_axis_off()
            ax.patch.set_facecolor('#222222')

        self.figure.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1
        )

        self.figure.set_size_inches(
            self.width() / self.figure.dpi,
            self.height() / self.figure.dpi
        )

        self.draw()
