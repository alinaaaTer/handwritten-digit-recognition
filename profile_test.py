import cProfile
from src.inference import load_model, predict_top3
import numpy as np

load_model()

x = np.random.rand(28, 28, 1).astype("float32")

cProfile.run("predict_top3(x)")