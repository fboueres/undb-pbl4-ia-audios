import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model

from config import (
    OUTPUTS_DIR,
    MODELS_DIR
)


def main():

    X = np.load(OUTPUTS_DIR / "X.npy")
    y = np.load(OUTPUTS_DIR / "y.npy")

    classes = np.load(
        OUTPUTS_DIR / "classes.npy",
        allow_pickle=True
    )

    history = np.load(
        OUTPUTS_DIR / "history.npy",
        allow_pickle=True
    ).item()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = load_model(
        MODELS_DIR / "sentiment_cnn.keras"
    )

    # =========================
    # Accuracy
    # =========================

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["accuracy"],
        label="Treino"
    )

    plt.plot(
        history["val_accuracy"],
        label="Validação"
    )

    plt.title("Accuracy por Época")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig(
        OUTPUTS_DIR / "training_accuracy.png"
    )

    plt.close()

    # =========================
    # Loss
    # =========================

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["loss"],
        label="Treino"
    )

    plt.plot(
        history["val_loss"],
        label="Validação"
    )

    plt.title("Loss por Época")
    plt.xlabel("Época")
    plt.ylabel("Loss")

    plt.legend()

    plt.savefig(
        OUTPUTS_DIR / "training_loss.png"
    )

    plt.close()

    # =========================
    # Matriz de confusão
    # =========================

    y_pred = model.predict(X_test)

    y_pred_classes = np.argmax(
        y_pred,
        axis=1
    )

    y_true_classes = np.argmax(
        y_test,
        axis=1
    )

    cm = confusion_matrix(
        y_true_classes,
        y_pred_classes
    )

    plt.figure(figsize=(8, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    display.plot()

    plt.title("Matriz de Confusão")

    plt.savefig(
        OUTPUTS_DIR / "confusion_matrix.png"
    )

    plt.close()

    print("Relatórios gerados com sucesso.")


if __name__ == "__main__":
    main()
