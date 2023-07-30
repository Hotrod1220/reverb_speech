import numpy as np
import struct
import wave

from pathlib import Path
from scipy.io import wavfile
from render import create

def main() -> None:
    dataset = Path.cwd().joinpath('original/rir')

    rir = [
        file
        for file in dataset.glob('*/*/*.wav')
        if file.is_file()
    ]

    speech = Path.cwd().joinpath('original.wav')

    sample_rir = rir[0]

    # Load the speech signal
    sample_rate_speech, speech_signal = wavfile.read(speech)

    # Padding the speech signal with zeros at the beginning
    padding = np.zeros((sample_rate_speech//2,), dtype=speech_signal.dtype)  # Padding with 0.5 seconds of zeros
    speech_signal_padded = np.concatenate([padding, speech_signal])

    # Write the padded speech signal to a new wav file
    padded_speech_path = Path.cwd().joinpath('padded_original.wav')
    wavfile.write(padded_speech_path, sample_rate_speech, speech_signal_padded)

    # Create the reverb
    reverb, header = create(padded_speech_path, sample_rir)

    # Trim the reverb to match the original length
    reverb = reverb[:len(speech_signal)]

    path = Path.cwd().joinpath('rendered.wav').as_posix()

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
