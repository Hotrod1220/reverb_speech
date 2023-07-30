from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import torch

from model import Model
from pathlib import Path
from prediction import Predictor


def main() -> None:
    current = Path.cwd()

    csv = current.joinpath('annotation.csv')
    annotation = pd.read_csv(csv)

    model = Model()
    model.device = 'cpu'

    state = torch.load('model/model.pth')
    model.load_state_dict(state)

    model.eval()

    mapping = {
        0: 'Large',
        1: 'Medium',
        2: 'Small'
    }

    predictor = Predictor()
    predictor.annotation = annotation
    predictor.mapping = mapping
    predictor.model = model

    sound = current.joinpath('sound')
    file = sound.joinpath('p232_085.wav')

    output = predictor.from_path(file)

    for filename in output:
        sample = output[filename]

        prediction = sample.get('prediction')
        label = prediction.get('label')

        print(label)

    plt.show()


if __name__ == '__main__':
    main()
