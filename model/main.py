import pandas as pd
import pickle
import torch

from dataset import ReverberationDataset
from model import Model
from pathlib import Path
from torch.utils.data import DataLoader
from trainer import Trainer
from transformation import Transformation


def main() -> None:
    root = Path.cwd().parent

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    torch.backends.cudnn.benchmark = True

    current = Path.cwd().joinpath('samples')

    path = root.joinpath('annotation.csv')

    if path.exists():
        annotation = pd.read_csv(path)
    else:
        room = [
            {
                'path': file.relative_to(current),
                'label': str(file.parent.relative_to(current))
            }
            for file in current.glob('*/*.wav')
            if file.is_file() and file.exists()
        ]

        annotation = pd.DataFrame.from_dict(room)
        annotation.to_csv('annotation.csv', index=False)

    hop_length = 512
    n_fft = 1024
    n_mels = 128
    sample, sample_rate = 16000, 16000

    settings = {
        'hop_length': hop_length,
        'n_fft': n_fft,
        'n_mels': n_mels,
        'sample_rate': sample_rate
    }

    transformation = Transformation(
        device=device,
        settings=settings
    )

    settings = {
        'sample': sample,
        'sample_rate': sample_rate
    }

    dataset = ReverberationDataset()
    dataset.annotation = annotation
    dataset.current = current
    dataset.device = device
    dataset.settings = settings
    dataset.transformation = transformation

    length = len(dataset)

    trl = int(length * 0.70)
    val = int(length * 0.15)
    tel = length - (trl + val)

    train, validation, test = torch.utils.data.random_split(
        dataset,
        [trl, val, tel]
    )

    batch_size = 128

    training = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True
    )

    testing = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False
    )

    validating = DataLoader(
        dataset=validation,
        batch_size=batch_size,
        shuffle=False
    )

    model = Model()
    model.device = device

    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.001
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        gamma=0.1,
        step_size=7
    )

    trainer = Trainer()
    trainer.device = device
    trainer.epoch = 10
    trainer.loss = loss
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.testing = testing
    trainer.training = training
    trainer.validating = validating
    trainer.start()

    torch.save(
        model.state_dict(),
        'model/model.pth'
    )

    with open('model/trainer.pkl', 'wb') as handle:
        pickle.dump(trainer, handle)


if __name__ == '__main__':
    main()
