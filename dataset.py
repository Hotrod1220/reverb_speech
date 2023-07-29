import pandas as pd
import torch
import torchaudio

from collections.abc import Callable
from pathlib import Path
from torch.utils.data import Dataset
from typing_extensions import Any


class ReverberationDataset(Dataset):
    def __init__(
        self,
        annotation: pd.DataFrame | None = None,
        current: Path | None = None,
        device: str | torch.device | None = None,
        settings: dict[str, Any] | None = None,
        transformation: Callable | None = None
    ):
        self.annotation = annotation
        self.current = current
        self.device = device
        self.mapping = {
            'large': 0,
            'medium': 1,
            'small': 2
        }
        self.settings = settings
        self.transformation = transformation

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.annotation.loc[index, 'path']
        label = self.annotation.loc[index, 'label']

        label = self.mapping.get(label)

        location = self.current.joinpath(path)

        signal, rate = torchaudio.load(location)
        signal = signal.to(self.device)
        signal = self.transformation(signal, rate)

        return signal, label
