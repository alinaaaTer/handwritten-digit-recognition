# app.py
import streamlit as st
import cv2
import numpy as np
import time

from src.preprocessing import preprocess_to_mnist
from src.inference import predict_top3, load_model
from src.io_validators import validate_uploaded_file
from src.logger import setup_logger
logger = setup_logger()

logger.info("Application started")

def run_app():
    """
    Run the Streamlit application for handwritten digit recognition.

    Functionality:
    - Upload image file (PNG/JPG)
    - Validate uploaded file
    - Preprocess image into MNIST format (28x28 grayscale)
    - Perform prediction using trained CNN model
    - Display prediction, confidence score, and top-3 results

    Returns:
        None
    """

    st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")
    st.title("Handwritten Digit Recognition (CNN, MNIST)")

    # Check if model is available
    try:
        load_model()
        logger.info("Model loaded successfully")
    except Exception:
        logger.error("Model loading failed")
        st.error("Model not found. First run: python train_cnn_mnist.py")
        st.stop()

    uploaded = st.file_uploader(
        "Upload image (PNG/JPG, max 5 MB)",
        type=["png", "jpg", "jpeg"]
    )

    invert = st.checkbox("Invert colors", value=True)

    # app.py
import streamlit as st
import cv2
import numpy as np

from src.preprocessing import preprocess_to_mnist
from src.inference import predict_top3, load_model
from src.io_validators import validate_uploaded_file
from src.logger import setup_logger
logger = setup_logger()

logger.info("Application started")

def run_app():
    """
    Run the Streamlit application for handwritten digit recognition.

    Functionality:
    - Upload image file (PNG/JPG)
    - Validate uploaded file
    - Preprocess image into MNIST format (28x28 grayscale)
    - Perform prediction using trained CNN model
    - Display prediction, confidence score, and top-3 results

    Returns:
        None
    """

    st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered")
    st.title("Handwritten Digit Recognition (CNN, MNIST)")

    # Check if model is available
    try:
        load_model()
        logger.info("Model loaded successfully")
    except Exception:
        logger.error("Model loading failed")
        st.error("Model not found. First run: python train_cnn_mnist.py")
        st.stop()

    uploaded = st.file_uploader(
        "Upload image (PNG/JPG, max 5 MB)",
        type=["png", "jpg", "jpeg"]
    )

    invert = st.checkbox("Invert colors", value=True)

    if uploaded:
      logger.info(f"File uploaded: {uploaded.name}")

      start = time.time()

      try:
          raw_bytes = uploaded.read()

          validate_uploaded_file(uploaded.name, raw_bytes)

          file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
          img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

          if img_bgr is None:
              raise ValueError("Image decoding failed")

          x = preprocess_to_mnist(img_bgr, invert=invert)

          st.subheader("Preprocessed (28×28)")
          st.image(x.squeeze(), clamp=True)

          pred, conf, top3 = predict_top3(x)

          logger.info(f"Prediction: {pred}, confidence: {conf}")

          end = time.time()
          logger.info(f"Processing time: {end - start:.4f} seconds")

          st.subheader("Result")
          st.write(f"**Predicted digit:** {pred}")
          st.write(f"**Confidence:** {conf:.3f}")
          st.write("**Top-3:**")

          for d, p in top3:
              st.write(f"- {d}: {p:.3f}")

      except Exception as e:
          from src.logger import log_error   # 👈 додай

          error_id = log_error(
              "File processing error",
              {"file": uploaded.name, "error": str(e)}
          )

          st.error(f"Error occurred. ID: {error_id}")


if __name__ == "__main__":
    run_app()