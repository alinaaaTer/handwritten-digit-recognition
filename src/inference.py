from pathlib import Path
import numpy as np
import tensorflow as tf

_MODEL = None


def load_model(model_path: str | Path = "models/cnn_mnist.h5"):
    """
    Load and cache the trained CNN model.

    This function loads the model from the specified path only once.
    Subsequent calls return the cached model instance.

    Args:
        model_path (str | Path): Path to the saved model file.

    Returns:
        tf.keras.Model: Loaded TensorFlow model.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model(str(model_path))
    return _MODEL


def predict_top3(x_28x28x1: np.ndarray):
    """
    Predict a digit from an input image and return top-3 predictions.

    The function expands the input image to match model input shape,
    performs inference, and returns the most probable digit along with
    confidence score and top-3 predictions.

    Args:
        x_28x28x1 (np.ndarray): Input image of shape (28, 28, 1),
            normalized to range [0, 1].

    Returns:
        tuple:
            - int: Predicted digit
            - float: Confidence score of the prediction
            - list[tuple[int, float]]: Top-3 predictions as (digit, probability)
    """
    model = load_model()

    x = np.expand_dims(x_28x28x1, axis=0)  # (1,28,28,1)
    probs = model.predict(x, verbose=0)[0]  # (10,)

    top_idx = np.argsort(probs)[::-1][:3]
    top3 = [(int(i), float(probs[i])) for i in top_idx]

    pred = top3[0][0]
    conf = top3[0][1]
    return pred, conf, top3