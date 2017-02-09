import numpy as np
import matplotlib.pyplot as plt

def load(fname, shape):
  with open(fname, "rb") as f:
    return np.reshape(np.fromstring(f.read(), dtype=np.uint8), shape, order="F")

