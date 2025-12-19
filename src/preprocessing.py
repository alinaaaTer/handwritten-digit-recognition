import cv2
import numpy as np

def preprocess_to_mnist(image_bgr: np.ndarray, invert: bool = True) -> np.ndarray:
    """
    Returns: np.ndarray shape (28, 28, 1) float32 in [0..1]
    """
    if image_bgr is None:
        raise ValueError("Empty image")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    if invert:
        resized = 255 - resized

    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)
    return x
