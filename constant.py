from __future__ import annotations

from pathlib import Path


def walk(file: Path) -> Path | None:
    for path in file.parents:
        if path.is_dir():
            venv = list(
                path.glob('venv')
            )

            for environment in venv:
                return environment.parent

            walk(path.parent)

    return None


file = Path.cwd()
CWD = walk(file).joinpath('render')

MODEL = CWD.joinpath('model')
DATASET = CWD.joinpath('dataset')

CONVERTED = DATASET.joinpath('converted')
ORIGINAL = DATASET.joinpath('original')
RENDER = DATASET.joinpath('render')

STATE = MODEL.joinpath('state')
