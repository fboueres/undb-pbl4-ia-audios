import numpy as np
import librosa

from config import (
    SAMPLE_RATE,
    N_MFCC,
    MAX_PAD_LEN
)

def extract_mfcc(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    audio = librosa.util.normalize(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    pad_width = MAX_PAD_LEN - mfcc.shape[1]

    if pad_width > 0:
        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    return mfcc
