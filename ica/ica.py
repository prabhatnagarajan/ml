from load import load
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
import scipy.io
from scipy.io.wavfile import write

def mix(inputs, A, sounds):
	U = np.zeros((len(inputs), len(sounds[0])))
	for i in range(len(inputs)):
		U[i] = sounds[inputs[i]]
	return np.dot(A, U)

def ica(X, n, learning_rate):
	m = np.shape(X)[0]
	#n x m matrix W
	W = np.zeros((n, m))
	#initialize W with random values
	for i in range(n):
		for j in range(m):
			W[i][j] = np.random.uniform(0, 0.1)
	for x in range(925):
		#Step 3 - Calculate Y=WX, Y is our current estimate of the source signals
		Y = W.dot(X)
		t = np.shape(Y)[1]
		#Step 4 - Calculate Z 
		Z = np.zeros((m, t))
		for i in range(n):
			for j in range(t):
				Z[i, j] = 1/(1 + np.exp(-Y[i,j]))
		#Step 5 - Calculate gradient
		gradient = learning_rate * np.dot(np.eye(n) + np.dot(np.full((m, t), 1.0) - 2 * Z, np.transpose(Y)), W)
		#Step 6
		W = W + gradient
	print W
	return Y

def main():
	#Step 1 - Load the data
	sounds = load("data/icaTest.mat", "U")
	print np.shape(sounds)
	A = load("data/icaTest.mat", "A")
	print np.shape(A)
	#Step 2 - mix the data
	#X = mix([0,1,2], A, sounds)
	X = np.dot(A, sounds)
	Y = ica(X, 3, 0.01)

def play(data, fname):
	scaled = np.int16(data/np.max(np.abs(data)) * 32767)
	write(fname, 44100, scaled)

if __name__ == '__main__':
	main()