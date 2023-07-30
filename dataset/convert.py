from __future__ import annotations

import soundfile as sf

from pathlib import Path
from tqdm import tqdm


def main() -> None:
    dataset = Path.cwd().joinpath('dataset')

    original = dataset.joinpath('original')
    converted = dataset.joinpath('converted')
    converted.mkdir(parents=True, exist_ok=True)

    glob = original.glob('*/*/*/*.flac')

    files = list(glob)
    total = len(files)

    for file in tqdm(files, total=total):
        filename = file.relative_to(original).with_suffix('.wav')

        path = converted.joinpath(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        signal, rate = sf.read(file)

        sf.write(
            path,
            signal,
            rate,
            format='wav',
            subtype='pcm_16'
        )


if __name__ == '__main__':
    main()
