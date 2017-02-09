import numpy as np
import matplotlib.pyplot as plt

def load(fname, shape):
  with open(fname, "rb") as f:
    return np.reshape(np.fromstring(f.read(), dtype=np.uint8), shape, order="F")

testImages = load("testImages.bin", (28, 28, 10000))
testLabels = load("testLabels.bin", (1, 10000))
trainImages = load("trainImages.bin", (28, 28, 60000))
trainLabels = load("trainLabels.bin", (1, 60000))

plt.imshow(trainImages[:, :, 1])
plt.show()