import numpy as np
import torch

from torch.nn.functional import pad
from torchaudio.transforms import MelSpectrogram, Resample
from typing import Any


class Transformation(torch.nn.Module):
    def __init__(
        self,
        device: str | torch.device | None = None,
        settings: dict[str, Any] = None
    ):
        super().__init__()

        self._mel = MelSpectrogram(**settings).to(device)

        self.device = device
        self.settings = settings

    def _cut(self, signal: torch.Tensor) -> torch.Tensor:
        target = self.settings.get('sample_rate')

        _, y = np.shape(signal)

        if y > target:
            signal = signal[:, :target]

        return signal

    def _mixdown(self, signal: torch.Tensor) -> torch.Tensor:
        x, _ = np.shape(signal)

        if x == 1:
            return signal

        signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _normalize(self, signal: torch.Tensor) -> torch.Tensor:
        mean, deviation = signal.mean(), signal.std()
        return (signal - mean) / deviation

    def _pad(self, signal: torch.Tensor) -> torch.Tensor:
        target = self.settings.get('sample_rate')

        _, y = np.shape(signal)

        if y < target:
            amount = target - y
            padding = (0, amount)

            signal = pad(signal, padding)

        return signal

    def _resample(self, signal: torch.Tensor, rate: int) -> torch.Tensor:
        target = self.settings.get('sample_rate')

        if rate == target:
            return signal

        resample = Resample(rate, target).to(self.device)
        signal = resample(signal)

        return signal

    def forward(self, x: torch.Tensor, rate: int) -> torch.Tensor:
        x = self._normalize(x)
        x = self._resample(x, rate)
        x = self._mixdown(x)
        x = self._cut(x)
        x = self._pad(x)
        x = self._mel(x)

        return x
