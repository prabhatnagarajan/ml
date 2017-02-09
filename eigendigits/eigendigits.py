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
def sample_images(testImages, num_samples):
	print np.shape(testImages)
	indices = np.random.randint(np.shape(testImages)[2], size=num_samples)
	print indices
	sample = np.zeros((28, 28, num_samples))
	for i in range(len(indices)):
		sample[:,:,i] = testImages[:,:,indices[i]]
	#print sample
	#print "that was sample"
	return sample

def vectorize_images(images):
	return np.reshape(images, (28 * 28, np.shape(images)[2]))

def hw1FindEigendigits(matrix):
	print "nothing"
	#vectorize_images()

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

def get_eigenvalues(matrix):
	num_rows = np.shape(matrix)[0]
	num_cols = np.shape(matrix)[1]
	eigen_info = np.linalg.eig(matrix)
	return eigen_info

def main():
	#10000 28 x 28 test images
	testImages = load("testImages.bin", (28, 28, 10000))
	#10000 labels
	testLabels = load("testLabels.bin", (1, 10000))
	#60000 28 x 28  images
	trainImages = load("trainImages.bin", (28, 28, 60000))
	# 60000 labels
	trainLabels = load("trainLabels.bin", (1, 60000))

	hw1FindEigendigits(trainImages)
	#plt.imshow(trainImages[:, :, 1])
	#plt.show()

	images = sample_images(testImages, 3)
	images = vectorize_images(images)
	cov = covariance_matrix(images)
	print cov
	eigen_info = get_eigenvalues(cov)
	print len(eigen_info)


if __name__ == '__main__':
	main()