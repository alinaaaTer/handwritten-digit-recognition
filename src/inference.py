from pathlib import Path
import numpy as np
import tensorflow as tf

_MODEL = None

def load_model(model_path: str | Path = "models/cnn_mnist.h5"):
    global _MODEL
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model(str(model_path))
    return _MODEL

def predict_top3(x_28x28x1: np.ndarray):
    """
    Input: shape (28,28,1) float32 [0..1]
    Output: predicted_digit, confidence, top3 list
    """
    model = load_model()

    x = np.expand_dims(x_28x28x1, axis=0)  # (1,28,28,1)
    probs = model.predict(x, verbose=0)[0]  # (10,)

    top_idx = np.argsort(probs)[::-1][:3]
    top3 = [(int(i), float(probs[i])) for i in top_idx]

    pred = top3[0][0]
    conf = top3[0][1]
    return pred, conf, top3
