import numpy as np
import matplotlib.pyplot as plt
from extract import *
from pdb import set_trace

filename = "data_GP/AG/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213203046-59968-right-speed_0.500.csv"
data = extract_data(filename)

def main():
	print data
	regression = []
	for i in data:
		regression.append(i+1)

	gradient_ascent(data[0::10],[0.1, 0.1, 0.1], np.array([0.1, 0.1, 0.1]))

	plt.plot(data)
	plt.plot(data + 1)
	plt.show()

def gradient_ascent(training_data, bounds, alphas):
	sigma_f, sigma_l, sigma_n = get_random_parameters(bounds[0], bounds[1], bounds[2])
	params = np.array([sigma_f, sigma_l, sigma_n])
	for i in range(100):
		gradient = np.array(compute_derivatives(training_data, params))
		params = params + np.multiply(alphas, gradient)
		print gradient
		print params
		set_trace()
	print "Final gradient is"
	gradient
	print "Final sigmas are"
	print params

def vector_tranpose(vector):
	return np.transpose(vector.reshape(len(vector),1))

def compute_derivatives(training_data, params):
	Q, delta_f, delta_l, delta_n = kernel(training_data, params[0], params[1], params[2])
	deltas = [delta_f, delta_l, delta_n]
	return [(0.5 * vector_tranpose(training_data).dot(np.linalg.inv(Q)).dot(delta).dot(np.linalg.inv(Q)).dot(training_data) - 0.5 * np.trace(np.linalg.inv(Q).dot(delta))) for delta in deltas]

#returns sigma of f, l, n
def get_random_parameters(bounda, boundb, boundc):
	return (np.random.uniform(-bounda, bounda), np.random.uniform(-boundb, boundb), np.random.uniform(-boundc, boundc))

def kernel(training_data, sigma_f, sigma_l, sigma_n):
	kernel = np.exp(sigma_f) * np.exp((-0.5) * np.exp(sigma_l) * np.outer(training_data, training_data))
	noise = np.exp(sigma_n) * np.eye(len(training_data))
	delta_f = kernel
	delta_l = np.dot(kernel, (-0.5) * np.exp(sigma_l) * np.outer(training_data, training_data))
	delta_n = noise
	return  (kernel + noise, delta_f, delta_l, delta_n)

if __name__ == '__main__':
	main()