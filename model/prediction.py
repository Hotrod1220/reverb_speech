from __future__ import annotations

import torch
import torchaudio

from model.transformation import Transformation
from typing import TYPE_CHECKING
from model.visualize import Envelope, Spectrogram

if TYPE_CHECKING:
    from model import Model
    from pathlib import Path
    from torch.utils.data import DataLoader
    from typing_extensions import Any


class Predictor():
    def __init__(
        self,
        device: str | torch.device = 'cpu',
        loader: DataLoader | None = None,
        mapping: dict[Any, Any] = None,
        model: Model = None,
        sample: None = None
    ):
        self.device = device
        self.loader = loader
        self.mapping = mapping
        self.model = model
        self.sample = sample

    def from_path(self, path: Path) -> list[dict[Any, Any]]:
        settings = {
            'hop_length': 512,
            'n_fft': 1024,
            'n_mels': 128,
            'sample_rate': 16000
        }

        transformation = Transformation(
            device=self.device,
            settings=settings
        )

        signal, rate = torchaudio.load(path)
        signal = signal.to(self.device)
        signal = transformation(signal, rate)

        signal = signal.unsqueeze(0)

        label = self.model(signal)

        label = label.detach().flatten()
        label = torch.argmax(label, dim=0).item()

        label = self.mapping[label]

        sample = {}

        sample[path.name] = {
            'original': {
                'envelope': Envelope.from_signal(path),
                'spectrogram': Spectrogram.from_signal(path, settings),
            },
            "transform": {
                'envelope': Envelope.from_tensor(signal),
                'spectrogram': Spectrogram.from_tensor(signal, settings),
            },
            'prediction': {
                'label': label,
            },
            'path': path
        }

        return sample
