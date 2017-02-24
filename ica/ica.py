from load import load
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
import scipy.io

def main():
	sounds = load("data/sounds.mat", "sounds")
	print np.shape(sounds)

if __name__ == '__main__':
	main()