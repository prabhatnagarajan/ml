import numpy as np
import matplotlib.pyplot as plt
from extract import *
import math
from pdb import set_trace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

filename = "data_GP/AG/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213203046-59968-right-speed_0.500.csv"
time, data = extract_data(filename)

def main():
	print data
	regression = []
	for i in data:
		regression.append(i+1)

	params = gradient_ascent(time[0::10], data[0::10],[0.1, 0.1, 0.03], np.array([0.005, 0.005, 0.0001]))
	mean, covariance = gpr(time[0::10], data[0::10], time[1::2], params)

	# kernel = 0.01 * RBF(length_scale = 0.05, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level = 0.0001, noise_level_bounds=(1e-10, 1e+1))
	# gp = GaussianProcessRegressor(kernel=kernel, alpha = 0.0).fit(time[0::10].reshape(-1,1), data[0::10].reshape(-1, 1))
	# mean, covariance = gp.predict(np.array(time[1::2]).reshape(-1, 1), return_cov = True)
	# mean = np.squeeze(mean)
	bottom = mean -  np.sqrt(np.diag(covariance))
	top = mean +  np.sqrt(np.diag(covariance))
	plt.plot(time, data, color="red")
	plt.plot(time[1::2], mean, 'k', lw=3, zorder=9)
	plt.fill_between(time[1::2], bottom, top, alpha=0.5, color='k')
	plt.title("Gaussian Process")
	plt.tight_layout()

	plt.show()


#Gets mean and covariance, similar to Marsland's algo in his GP Chapter
def gpr(time, data, test_times, params):
	K = kernel(time, params[0], params[1], params[2])[0]
	kstar = kernel2(time, test_times, params[0], params[1], params[2])
	kstarstar = kernel2(test_times, test_times, params[0], params[1], params[2])
	mean = np.transpose(kstar).dot(np.linalg.inv(K)).dot(data)
	covariance = kstarstar - np.transpose(kstar).dot(np.linalg.inv(K)).dot(kstar)
	return (mean, covariance)

def k(x1, x2, sigma_f, l):
	return math.pow(sigma_f, 2) * np.exp((-1) * (1/(2 * l * l)) * math.pow(x1 - x2, 2))

def gradient_ascent(time, training_data, bounds, alphas):
	sigma_f, sigma_l, sigma_n = get_random_parameters(bounds[0], bounds[1], bounds[2])
	params = np.array([sigma_f, sigma_l, sigma_n])
	for i in range(500):
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

def kernel2(time, time2, sigma_f, sigma_l, sigma_n):
	kernel = np.exp(sigma_f) * np.exp((-0.5) * np.exp(sigma_l) * get_sq_diff_mat2(time, time2))
	return  kernel

def get_sq_diff_mat2(time, time2):
	mat = np.zeros((len(time), len(time2)))
	for i in range(len(time)):
		for j in range(len(time2)):
			mat[i,j] = math.pow(time[i] - time2[j], 2)
	return mat

if __name__ == '__main__':
	main()