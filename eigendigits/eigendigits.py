from load import load
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

'''
Takes in X x k matrix A, X number of pixels, and k
is the number of training images

Returns: 1) vector m of length x containing the mean column
vector of A,  
2) An (X by k) matrix V that contains k 
eigenvectors of the covariance matrix of A (after the mean has been subtracted). 
-Columns of V are sorted in descending order by eigenvalue
'''
def sample_images(images, num_samples):
	indices = np.random.randint(np.shape(images)[2], size=num_samples)
	sample = np.zeros((28, 28, num_samples))
	for i in range(len(indices)):
		sample[:,:,i] = images[:,:,indices[i]]
	#print sample
	#print "that was sample"
	return sample

def vectorize_images(images):
	return np.reshape(images, (28 * 28, np.shape(images)[2]))

def hw1FindEigendigits(matrix):
	new_mat = mean_subtract(matrix)
	eigen_info = eigenvalues(a_transpose_a(new_mat))
	eigenvectors = []
	for vector in eigen_info[1]:
		eigenvectors.append(np.dot(new_mat, vector))
	eigenmap = dict()
	for i in range(len(eigen_info[0])):
		if eigen_info[0][i] in eigenmap:
			eigenmap[eigen_info[0][i]].append(eigen_info[1][i])
		else:
			vectors = list()
			vectors.append(eigen_info[1][i])
			eigenmap[eigen_info[0][i]] = vectors
	sorted_eigenvals = sorted(eigenmap.keys())
	print sorted_eigenvals
	return (eigen_info[0], eigenvectors)

def mean_column(matrix):
	return np.mean(matrix, axis=1)

def mean_subtract(matrix):
	mean = mean_column(matrix)
	matrix_sub = np.zeros(np.shape(matrix))
	for i in range(np.shape(matrix)[1]):
		matrix_sub[:,i] = matrix[:,i] - mean
	return matrix_sub

def covariance_matrix(matrix):
	mat = mean_subtract(matrix)
	return np.dot(mat, mat.transpose())/np.shape(matrix)[0]

def eigenvalues(matrix):
	eigen_info = np.linalg.eig(matrix)
	return eigen_info

def get_eigenvalues(matrix):
	mat =  np.dot(matrix.transpose(), matrix)
	eigen_info = np.linalg.eig(np.dot(matrix.transpose(),matrix))
	return eigen_info

def a_transpose_a(matrix):
	return np.dot(matrix.transpose(),matrix)

def main():
	#10000 28 x 28 test images
	testImages = load("testImages.bin", (28, 28, 10000))
	#10000 labels
	testLabels = load("testLabels.bin", (1, 10000))
	#60000 28 x 28  images
	trainImages = load("trainImages.bin", (28, 28, 60000))
	# 60000 labels
	trainLabels = load("trainLabels.bin", (1, 60000))

	images = sample_images(trainImages, 5)
	images = vectorize_images(images)
	#cov = covariance_matrix(images)
	#NOTE: Maybe subtract mean before getting eigenvalues
	eigen_info = hw1FindEigendigits(images)
	print eigen_info[0]
	for i in range(len(eigen_info[0]) - 1):
		print eigen_info[0][i] >= eigen_info[0][i + 1]
	#hw1FindEigendigits(trainImages)
	#plt.imshow(trainImages[:, :, 1])
	#plt.show()

if __name__ == '__main__':
	main()