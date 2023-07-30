from __future__ import annotations

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import hilbert
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pathlib import Path
    from typing_extensions import Any, Self


class Envelope():
    @classmethod
    def from_signal(cls: type[Self], path: Path | npt.NDArray) -> tuple[Figure, Axes]:
        signal, _ = librosa.load(path)

        analytic = hilbert(signal)
        envelope = np.abs(analytic)

        figsize = (18, 4)
        fig, ax = plt.subplots(figsize=figsize)

        start = 0
        end = 500

        ax.plot(
            envelope[start:end],
            linewidth=1.0,
            fillstyle='none'
        )

        plt.title('Envelope for Original')
        plt.tight_layout()

        return fig, ax

    @classmethod
    def from_tensor(cls: type[Self], tensor: npt.NDArray) -> tuple[Figure, Axes]:
        signal = tensor.squeeze().cpu().numpy()

        analytic = hilbert(signal)
        envelope = np.abs(analytic)

        figsize = (18, 4)
        fig, ax = plt.subplots(figsize=figsize)

        start = 0
        end = 500

        ax.plot(
            envelope[start:end],
            linewidth=1.0,
            fillstyle='none'
        )

        plt.title('Envelope for Transform')
        plt.tight_layout()

        return fig, ax


class Spectrogram():
    @classmethod
    def from_signal(
        cls: type[Self],
        path: Path,
        settings: dict[str, Any]
    ) -> tuple[Figure, Axes]:
        n_fft = settings.get('n_fft')
        hop_length = settings.get('hop_length')
        sr = settings.get('sample_rate')
        fmin = 0
        fmax = sr / 2

        signal, sr = librosa.load(path)

        spectrogram = librosa.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length
        )

        spectrogram = librosa.amplitude_to_db(
            np.abs(spectrogram),
            ref=np.max
        )

        figsize = (18, 4)
        fig, ax = plt.subplots(figsize=figsize)

        librosa.display.specshow(
            spectrogram,
            sr=sr,
            ax=ax,
            fmin=fmin,
            fmax=fmax,
            hop_length=hop_length,
            n_fft=n_fft,
            x_axis='time',
        )

        plt.title('Spectrogram')
        plt.tight_layout()

        return fig, ax

    @classmethod
    def from_tensor(
        cls: type[Self],
        tensor: npt.NDArray,
        settings: dict[str, Any]
    ) -> tuple[Figure, Axes]:
        n_fft = settings.get('n_fft')
        hop_length = settings.get('hop_length')
        sr = settings.get('sample_rate')
        fmin = 0
        fmax = sr / 2

        signal = tensor.squeeze().cpu().numpy()

        figsize = (18, 4)
        fig, ax = plt.subplots(figsize=figsize)

        librosa.display.specshow(
            signal,
            sr=sr,
            ax=ax,
            fmin=fmin,
            fmax=fmax,
            hop_length=hop_length,
            n_fft=n_fft,
            x_axis='time',
        )

        plt.title('Mel')
        plt.tight_layout()

        return fig, ax
