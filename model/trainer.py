from __future__ import annotations

import torch

from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model import Model
    from torch.nn import CrossEntropyLoss
    from torch.optim.adam import Adam
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data.dataloader import DataLoader


class Trainer():
    def __init__(
        self,
        device: str | torch.device = 'cpu',
        epoch: int = 0,
        loss: CrossEntropyLoss = None,
        model: Model = None,
        optimizer: Adam = None,
        scheduler: StepLR = None,
        testing: DataLoader = None,
        training: DataLoader = None,
        validating: DataLoader = None,

    ):
        self._model = model

        self.device = device
        self.epoch = epoch
        self.loss = loss
        self.scheduler = scheduler
        self.testing = testing
        self.training = training
        self.validating = validating
        self.optimizer = optimizer

    def _single_validation_epoch(self) -> tuple[float, float]:
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0

        total = len(self.validating)
        progress = tqdm(self.validating, total=total)

        for x, y in progress:
            x = x.to(self.device)
            y = y.to(self.device)

            logit = self.model(x)

            output = self.loss(logit, y)

            total_loss = total_loss + output.item()

            _, prediction = torch.max(logit, dim=1)

            correct = torch.sum(prediction == y).item()
            total_accuracy = total_accuracy + correct

        return (
            total_loss / len(self.validating),
            total_accuracy / len(self.validating.dataset)
        )

    def _single_training_epoch(self) -> None:
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0

        total = len(self.training)
        progress = tqdm(self.training, total=total)

        for x, y in progress:
            x = x.to(self.device)
            y = y.to(self.device)

            logit = self.model(x)

            output = self.loss(logit, y)

            total_loss = total_loss + output.item()

            _, prediction = torch.max(logit, dim=1)

            correct = torch.sum(prediction == y).item()
            total_accuracy = total_accuracy + correct

            output.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.scheduler.step()

        return (
            total_loss / len(self.training),
            total_accuracy / len(self.training.dataset)
        )

    def start(self) -> None:
        self.model.to(self.device)

        for i in range(self.epoch):
            print(f"[Epoch {i + 1}]")

            loss, accuracy = self._single_training_epoch()

            print(f"training_accuracy: {accuracy:.4f}")
            print(f"training_loss: {loss:.4f}")

            loss, accuracy = self._single_validation_epoch()

            print(f"validation_accuracy: {accuracy:.4f}")
            print(f"validation_loss: {loss:.4f}")

            print()

        print('Training is complete')
