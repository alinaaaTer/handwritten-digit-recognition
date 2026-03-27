Handwritten Digit & Letter Recognition

This project is a software system for automatic recognition of handwritten digits and letters with further use of results in electronic forms.

The system is designed to speed up data entry during the digitization of handwritten documents such as questionnaires, tests, receipts, and other forms used in educational, medical, and administrative institutions.

The project is implemented as a multi-component system developed step by step.

General Functionality
Recognition of handwritten digits and (optionally) letters from images
Image preprocessing to improve recognition quality
Use of CNN-based models for classification
Comparison of different model architectures
Analysis of data augmentation impact on accuracy
Evaluation on standard datasets and custom samples
Output of prediction results including:
recognized symbol
confidence score
top-3 alternative predictions

Technologies
Python
TensorFlow / PyTorch (depending on implementation)
NumPy
OpenCV / PIL

## Code Documentation

All functions must include docstrings.

Each docstring should contain:
- description
- parameters
- return values

## Getting Started

### 1. Clone repository

git clone https://github.com/alinaaaTer/handwritten-digit-recognition
cd handwritten-digit-recognition

### 2. Install Python

Install Python 3.10+

### 3. Create virtual environment

python -m venv .venv

### 4. Activate environment

Windows:
.venv\Scripts\activate

### 5. Install dependencies

pip install -r requirements.txt

### 6. Train model

python train_cnn_mnist.py

### 7. Run application

streamlit run app.py

### 8. Open in browser

http://127.0.0.1:5500/

