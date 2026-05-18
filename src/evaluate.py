import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

from config import OUTPUTS_DIR, MODELS_DIR


def main():
    X = np.load(OUTPUTS_DIR / "X.npy")
    y = np.load(OUTPUTS_DIR / "y.npy")
    classes = np.load(OUTPUTS_DIR / "classes.npy", allow_pickle=True)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = load_model(MODELS_DIR / "sentiment_cnn.keras")

    y_pred = model.predict(X_test)

    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("Relatório de classificação:")
    print(
        classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=classes
        )
    )

    cm = confusion_matrix(y_true_classes, y_pred_classes)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    print(cm)

    print("Matriz de confusão salva em outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
