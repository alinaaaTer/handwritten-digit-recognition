# app.py
import streamlit as st
import cv2
import numpy as np

from src.preprocessing import preprocess_to_mnist
from src.inference import predict_top3, load_model
from src.io_validators import validate_uploaded_file

st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")
st.title("Handwritten Digit Recognition (CNN, MNIST)")

# Перевірка, що модель існує/завантажується
try:
    load_model()
except Exception:
    st.error("Model not found. First run: python train_cnn_mnist.py")
    st.stop()

uploaded = st.file_uploader("Upload image (PNG/JPG, max 5 MB)", type=["png", "jpg", "jpeg"])
invert = st.checkbox("Invert colors", value=True)

if uploaded:
    raw_bytes = uploaded.read()

    try:
        validate_uploaded_file(uploaded.name, raw_bytes)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Cannot decode image. Upload a valid PNG/JPG.")
        st.stop()

    x = preprocess_to_mnist(img_bgr, invert=invert)

    st.subheader("Preprocessed (28×28)")
    st.image(x.squeeze(), clamp=True)

    pred, conf, top3 = predict_top3(x)

    st.subheader("Result")
    st.write(f"**Predicted digit:** {pred}")
    st.write(f"**Confidence:** {conf:.3f}")
    st.write("**Top-3:**")
    for d, p in top3:
        st.write(f"- {d}: {p:.3f}")
