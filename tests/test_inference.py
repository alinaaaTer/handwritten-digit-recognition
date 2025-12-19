import numpy as np
import pytest

from src import inference


class DummyModel:
    """
    Заглушка нейронної мережі для unit-тестів.
    Повертає контрольований вектор ймовірностей
    """
    def __init__(self, probs):
        self.probs = np.array(probs, dtype=float)

    def predict(self, x, verbose=0):
        # Перевіряємо контракт: функція повинна подавати (1, 28, 28, 1)
        assert x.shape == (1, 28, 28, 1)
        return np.array([self.probs], dtype=float)


def test_predict_top3_returns_expected_pred_and_conf(monkeypatch):
    """
    TC-08: Перевірка, що predict_top3 повертає правильний pred та confidence.
    """
    probs = np.zeros(10)
    probs[7] = 0.85  # найімовірніший клас — 7

    dummy = DummyModel(probs)

    # Замінюємо реальне завантаження моделі на заглушку
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: dummy)

    x = np.random.rand(28, 28, 1).astype(np.float32)
    pred, conf, top3 = inference.predict_top3(x)

    assert pred == 7
    assert conf == pytest.approx(0.85, rel=1e-6)
    assert len(top3) == 3


def test_predict_top3_top3_is_sorted_desc(monkeypatch):
    """
    TC-09: Перевірка, що top-3 відсортований за спаданням ймовірності.
    """
    probs = np.array([0.01, 0.50, 0.20, 0.10, 0.05,
                      0.02, 0.03, 0.04, 0.01, 0.04])

    dummy = DummyModel(probs)
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: dummy)

    x = np.random.rand(28, 28, 1).astype(np.float32)
    _, _, top3 = inference.predict_top3(x)

    top_probs = [p for _, p in top3]
    assert top_probs[0] >= top_probs[1] >= top_probs[2]


def test_predict_top3_returns_int_digit_and_conf_in_range(monkeypatch):
    """
    TC-10: Перевірка типів і діапазону значень.
    pred має бути int, confidence має бути в межах [0;1].
    """
    probs = np.zeros(10)
    probs[3] = 0.99

    dummy = DummyModel(probs)
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: dummy)

    x = np.random.rand(28, 28, 1).astype(np.float32)
    pred, conf, _ = inference.predict_top3(x)

    assert isinstance(pred, int)
    assert 0.0 <= conf <= 1.0


def test_predict_top3_rejects_wrong_shape(monkeypatch):
    """
    TC-11: Негативний сценарій — некоректна форма вхідних даних.
    Очікуємо, що функція впаде з помилкою.
    """
    probs = np.zeros(10)
    probs[1] = 0.9

    dummy = DummyModel(probs)
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: dummy)

    bad_x = np.random.rand(28, 28).astype(np.float32) 

    with pytest.raises(Exception):
        inference.predict_top3(bad_x)
