from pathlib import Path
import tensorflow as tf

MODEL_PATH = Path("models/cnn_mnist.h5")


def build_model():
    """
    Build and compile a Convolutional Neural Network (CNN) for MNIST digit classification.

    The model consists of convolutional, pooling, and fully connected layers.
    It is compiled using the Adam optimizer and sparse categorical crossentropy loss.

    Returns:
        tf.keras.Model: Compiled CNN model ready for training.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    """
    Train a CNN model on the MNIST dataset and save it to disk.

    Workflow:
    - Load MNIST dataset
    - Normalize and reshape input data
    - Train the model with validation split
    - Evaluate model performance on test data
    - Save trained model to file

    Returns:
        None
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype("float32")[..., None]
    x_test = (x_test / 255.0).astype("float32")[..., None]

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        )
    ]

    model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=8,
        batch_size=128,
        callbacks=callbacks,
        verbose=2
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
