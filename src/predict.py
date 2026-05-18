import sys
import numpy as np

from tensorflow.keras.models import load_model

from preprocess import extract_mfcc
from config import (
    MODELS_DIR,
    OUTPUTS_DIR
)


def predict(file_path):
    model = load_model(
        MODELS_DIR / "sentiment_cnn.keras"
    )

    classes = np.load(
        OUTPUTS_DIR / "classes.npy",
        allow_pickle=True
    )

    mfcc = extract_mfcc(file_path)

    X = np.expand_dims(mfcc, axis=0)
    X = X[..., np.newaxis]

    prediction = model.predict(X)

    predicted_class = np.argmax(prediction)

    sentiment = classes[predicted_class]

    confidence = prediction[0][predicted_class]

    print(f"Sentimento previsto: {sentiment}")
    print(f"Confiança: {confidence:.2%}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Uso:")
        print("python src/predict.py audio.wav")
        sys.exit()

    audio_file = sys.argv[1]

    predict(audio_file)
