from __future__ import annotations

import numpy as np
import struct
import wave

from pathlib import Path
from scipy.signal import convolve
from secrets import SystemRandom
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from typing_extensions import Any


def create(sin: Path, rin: Path) -> tuple[npt.NDArray, tuple[Any, Any]]:
    sin = sin.as_posix()
    rin = rin.as_posix()

    # Get information about the source file
    file = wave.open(sin, 'r')
    rate = file.getframerate()
    nsample_source = file.getnframes()
    frames_source = file.readframes(nsample_source)
    file.close()

    # Get information about the room impulse response
    file = wave.open(rin, 'r')
    nsample_reverb = file.getnframes()
    frames_reverb = file.readframes(nsample_reverb)
    file.close()

    maximum = np.iinfo(np.int16).max

    # Unpack the source file
    source = struct.unpack(
        f"{nsample_source}h",
        frames_source
    )

    source = np.array(source, dtype=np.float64)
    source = source / maximum

    # Unpack the room impulse response
    reverb = struct.unpack(
        f"{nsample_reverb}h",
        frames_reverb
    )

    reverb = np.array(reverb, dtype=np.float64)
    reverb = reverb / maximum

    gain = 1

    # Convolve
    output = convolve(source, reverb, mode='full') * gain

    output = output / np.max(
        np.abs(output)
    )

    render = output * int(maximum)
    render = render.astype(np.int16)

    nframes = len(render)
    comptype = 'NONE'
    compname = 'not compressed'
    nchannels = 1
    sampwidth = 2

    header = (
        nchannels,
        sampwidth,
        rate,
        nframes,
        comptype,
        compname
    )

    return render, header


def main() -> None:
    dataset = Path.cwd().joinpath('dataset')
    path = dataset.joinpath('render')

    speech_small = path.joinpath('small')
    speech_small.mkdir(exist_ok=True, parents=True)

    speech_medium = path.joinpath('medium')
    speech_medium.mkdir(exist_ok=True, parents=True)

    speech_large = path.joinpath('large')
    speech_large.mkdir(exist_ok=True, parents=True)

    speech = [
        file
        for file in dataset.joinpath('converted/train-clean-100').glob('*/*/*.wav')
        if file.is_file()
    ]

    rir_small = [
        file
        for file in dataset.joinpath('original/rir/smallroom').glob('*/*.wav')
        if file.is_file()
    ]

    rir_medium = [
        file
        for file in dataset.joinpath('original/rir/mediumroom').glob('*/*.wav')
        if file.is_file()
    ]

    rir_large = [
        file
        for file in dataset.joinpath('original/rir/largeroom').glob('*/*.wav')
        if file.is_file()
    ]

    generator = SystemRandom()

    total = len(speech)

    for file in tqdm(speech, total=total):
        small = generator.sample(rir_small, 1)
        medium = generator.sample(rir_medium, 1)
        large = generator.sample(rir_large, 1)

        for s, m, l in zip(small, medium, large):
            # Small
            reverb, header = create(file, s)

            path = speech_small.joinpath(file.name).as_posix()

            wav = wave.open(path, 'w')
            wav.setparams(header)

            length = len(reverb)

            wav.writeframes(
                struct.pack(
                    f"{length}h",
                    *reverb
                )
            )

            wav.close()

            # Medium
            reverb, header = create(file, m)

            path = speech_medium.joinpath(file.name).as_posix()

            wav = wave.open(path, 'w')
            wav.setparams(header)

            length = len(reverb)

            wav.writeframes(
                struct.pack(
                    f"{length}h",
                    *reverb
                )
            )

            wav.close()

            # Large
            reverb, header = create(file, l)

            path = speech_large.joinpath(file.name).as_posix()

            wav = wave.open(path, 'w')
            wav.setparams(header)

            length = len(reverb)

            wav.writeframes(
                struct.pack(
                    f"{length}h",
                    *reverb
                )
            )

            wav.close()


if __name__ == '__main__':
    main()
