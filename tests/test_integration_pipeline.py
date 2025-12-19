import numpy as np
import pytest
import cv2

from src.io_validators import validate_uploaded_file
from src.preprocessing import preprocess_to_mnist
from src import inference


class DummyModel:
    """Заглушка моделі: повертає контрольовані ймовірності."""
    def __init__(self, probs):
        self.probs = np.array(probs, dtype=float)

    def predict(self, x, verbose=0):
        # Контракт: модель має отримати (1, 28, 28, 1)
        assert x.shape == (1, 28, 28, 1)
        return np.array([self.probs], dtype=float)


def make_valid_png_bytes() -> bytes:
    """Створює валідне PNG в памʼяті (без файлів на диску)."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.putText(img, "7", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def decode_bytes_to_bgr(raw_bytes: bytes):
    """Імітація того, що робить app.py: bytes -> cv2.imdecode -> BGR."""
    file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def test_it01_full_pipeline_valid_image(monkeypatch):
    """
    IT-01: validators -> decode -> preprocess -> inference.
    Перевіряємо взаємодію модулів і контракт даних між ними.
    """
    raw = make_valid_png_bytes()

    # 1) Валідація вхідного файлу (ім'я + байти)
    validate_uploaded_file("digit.png", raw)

    # 2) Decode
    img_bgr = decode_bytes_to_bgr(raw)
    assert img_bgr is not None

    # 3) Preprocess до MNIST формату
    x = preprocess_to_mnist(img_bgr, invert=True)
    assert x.shape == (28, 28, 1)
    assert x.dtype == np.float32
    assert 0.0 <= float(x.min()) <= 1.0
    assert 0.0 <= float(x.max()) <= 1.0

    # 4) Inference (модель замінюємо заглушкою)
    probs = np.zeros(10)
    probs[7] = 0.85
    monkeypatch.setattr(inference, "load_model", lambda *a, **k: DummyModel(probs))

    pred, conf, top3 = inference.predict_top3(x)

    assert pred == 7
    assert 0.0 <= conf <= 1.0
    assert len(top3) == 3
    # top-3 має бути відсортований за спаданням
    top_probs = [p for _, p in top3]
    assert top_probs[0] >= top_probs[1] >= top_probs[2]


def test_it02_invalid_extension_blocked_by_validator():
    """
    IT-02 (негативний): validators блокує пайплайн на етапі формату файлу.
    """
    raw = make_valid_png_bytes()
    with pytest.raises(ValueError):
        validate_uploaded_file("file.pdf", raw)


def test_it03_corrupted_bytes_not_decodable():
    """
    IT-03 (негативний): bytes не декодуються OpenCV -> img_bgr=None.
    Це інтеграція "decode етапу" як в app.py.
    """
    raw = b"not an image"
    try:
        validate_uploaded_file("bad.jpg", raw)
    except ValueError:
        return

    img_bgr = decode_bytes_to_bgr(raw)
    assert img_bgr is None
