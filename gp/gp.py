import numpy as np
import matplotlib.pyplot as plt
from extract import *
import math
from pdb import set_trace


filename = "data_GP/AG/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213203046-59968-right-speed_0.500.csv"
time, data = extract_data(filename)

def main():
	print data
	regression = []
	for i in data:
		regression.append(i+1)

	params = gradient_ascent(time[0::10], data[0::10],[0.1, 0.1, 0.03], np.array([0.005, 0.005, 0.0001]))


	plt.plot(time, data)
	# plt.plot(data + 1)
	plt.show()

# #Gets mean and covariance, adapted from Marsland's code in his GP Chapter
# def gpr(time, data, test_times, params):

def k(x1, x2, sigma_f, l):
	return math.pow(sigma_f, 2) * np.exp((-1) * (1/(2 * l * l)) * math.pow(x1 - x2, 2))

def gradient_ascent(time, training_data, bounds, alphas):
	sigma_f, sigma_l, sigma_n = get_random_parameters(bounds[0], bounds[1], bounds[2])
	params = np.array([sigma_f, sigma_l, sigma_n])
	for i in range(5000):
		gradient = compute_derivatives(time, training_data, params)
		params = params + alphas *  gradient
		if i % 1000 == 0:
			print "Done " + str(i)
		# print gradient
		# print params
		#set_trace()
	print "Final gradient is"
	print gradient
	print "Final sigmas are"
	print params
	# new_params = []
	# new_params.append(np.exp(params[0]/2))
	# new_params.append(np.exp(-params[1]/2))
	# new_params.append(np.exp(params[2]/2))
	return params

def get_sq_diff_mat(time):
	mat = np.zeros((len(time), len(time)))
	for i in range(len(time)):
		for j in range(len(time)):
			mat[i,j] = math.pow(time[i] - time[j], 2)
	return mat

def vector_tranpose(vector):
	return np.transpose(vector.reshape(len(vector),1))

def compute_derivatives(time, training_data, params):
	Q, delta_f, delta_l, delta_n = kernel(time, params[0], params[1], params[2])
	deltas = [delta_f, delta_l, delta_n]
	return [float(0.5 * vector_tranpose(training_data).dot(np.linalg.inv(Q)).dot(delta).dot(np.linalg.inv(Q)).dot(training_data) - 0.5 * np.trace(np.linalg.inv(Q).dot(delta))) for delta in deltas]

#returns sigma of f, l, n
def get_random_parameters(bounda, boundb, boundc):
	return (np.random.uniform(-bounda, bounda), np.random.uniform(-boundb, boundb), np.random.uniform(-boundc, boundc))

def kernel(time, sigma_f, sigma_l, sigma_n):
	kernel = np.exp(sigma_f) * np.exp((-0.5) * np.exp(sigma_l) * get_sq_diff_mat(time))
	noise = np.exp(sigma_n) * np.eye(len(time))
	delta_f = kernel
	delta_l = np.dot(kernel, (-0.5) * np.exp(sigma_l) * get_sq_diff_mat(time))
	delta_n = noise
	return  (kernel + noise, delta_f, delta_l, delta_n)

if __name__ == '__main__':
	main()