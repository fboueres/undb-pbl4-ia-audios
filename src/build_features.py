import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from load_dataset import load_audio_metadata
from preprocess import extract_mfcc
from config import OUTPUTS_DIR


def main():
    df = load_audio_metadata()

    X = []
    y = []

    for index, row in df.iterrows():
        mfcc = extract_mfcc(row["file_path"])

        X.append(mfcc)
        y.append(row["sentiment"])

        if (index + 1) % 500 == 0:
            print(f"{index + 1} áudios processados...")

    X = np.array(X)
    y = np.array(y)

    X = X[..., np.newaxis]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    OUTPUTS_DIR.mkdir(exist_ok=True)

    np.save(OUTPUTS_DIR / "X.npy", X)
    np.save(OUTPUTS_DIR / "y.npy", y_categorical)
    np.save(OUTPUTS_DIR / "classes.npy", encoder.classes_)

    print("Features salvas com sucesso.")
    print("X shape:", X.shape)
    print("y shape:", y_categorical.shape)
    print("Classes:", encoder.classes_)


if __name__ == "__main__":
    main()
