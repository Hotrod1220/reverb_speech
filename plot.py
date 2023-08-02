from __future__ import annotations

import matplotlib.pyplot as plt
import pickle

from pathlib import Path


def main() -> None:
    path = Path.cwd()
    path = path.joinpath('model/state/history.pkl')

    with open(path, 'rb') as handle:
        history = pickle.load(handle)

    training = history.get('training')
    validation = history.get('validation')

    # Accuracy
    tc_loss = training.get('classification_accuracy')
    vc_loss = validation.get('classification_accuracy')

    figsize = (10, 5)
    plt.figure(figsize=figsize)

    plt.plot(
        tc_loss,
        label='Training Accuracy'
    )

    plt.plot(
        vc_loss,
        label='Validation Accuracy'
    )

    plt.title('Classification: Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(
        'accuracy.png',
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    plt.show()

    # Loss
    tc_loss = training.get('classification_loss')
    vc_loss = validation.get('classification_loss')

    figsize = (10, 5)
    plt.figure(figsize=figsize)

    plt.plot(
        tc_loss,
        label='Training Loss'
    )

    plt.plot(
        vc_loss,
        label='Validation Loss'
    )

    plt.title('Classification: Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(
        'loss.png',
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    plt.show()


if __name__ == '__main__':
    main()
