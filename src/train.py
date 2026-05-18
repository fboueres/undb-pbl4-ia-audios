import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from config import OUTPUTS_DIR, MODELS_DIR


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    X = np.load(OUTPUTS_DIR / "X.npy")
    y = np.load(OUTPUTS_DIR / "y.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_model(
        input_shape=X_train.shape[1:],
        num_classes=y_train.shape[1]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    y_train_labels = np.argmax(y_train, axis=1)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )

    class_weights = dict(enumerate(class_weights))

    print("Pesos das classes:")
    print(class_weights)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping],
        class_weight=class_weights
    )

    loss, accuracy = model.evaluate(X_test, y_test)

    print(f"Loss teste: {loss:.4f}")
    print(f"Acurácia teste: {accuracy:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)

    model.save(MODELS_DIR / "sentiment_cnn.keras")

    print("Modelo salvo em models/sentiment_cnn.keras")


if __name__ == "__main__":
    main()
