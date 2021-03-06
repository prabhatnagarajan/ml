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
def sample_images(images, labels, num_samples):
	indices = np.random.randint(np.shape(images)[2], size=num_samples)
	sample = np.zeros((28, 28, num_samples))
	label = np.zeros(num_samples, dtype=np.int8)
	for i in range(len(indices)):
		sample[:,:,i] = images[:,:,indices[i]]
		label[i] = labels[0][indices[i]]
	#print sample
	#print "that was sample"
	return (sample, label)

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
			eigenmap[eigen_info[0][i]].append(eigenvectors[i])
		else:
			vectors = list()
			vectors.append(eigenvectors[i])
			eigenmap[eigen_info[0][i]] = vectors
	sorted_eigenvals = sorted(eigenmap.keys())
	sorted_eigenvals.reverse()
	sorted_eigenvectors = np.zeros((len(eigenvectors[0]), len(eigenvectors)))
	count = 0
	for i in range(len(sorted_eigenvals)):
		for k in range(len(eigenmap[sorted_eigenvals[i]])):
			sorted_eigenvectors[:, count] = eigenmap[sorted_eigenvals[i]][k]
			count = count + 1
	return (mean_column(matrix), sorted_eigenvectors)

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

def project(vector, mean, matrix):
	return np.dot(matrix.transpose(), vector - mean)

def unproject(vector, mean, matrix):
	return np.dot(np.linalg.pinv(matrix.transpose()), vector) + mean

def project_vectors(vectors, mean, matrix):
	num_vectors = np.shape(vectors)[1]
	projection = np.zeros((np.shape(matrix)[1], num_vectors))
	for i in range(num_vectors):
		projection[:, i] = project(vectors[:, i], mean, matrix)
	return projection

'''
returns labels
'''
def kNN(K, sample_projections, sample_labels, test_projections):
	labels = []
	for i in range(np.shape(test_projections)[1]):
		test_vector = test_projections[:, i]
		distances = []
		for i in range(np.shape(sample_projections)[1]):
			train_vector = sample_projections[:, i]
			dist = np.linalg.norm(test_vector - train_vector)
			distances.append((dist, i))
		distances = sorted(distances)
		#get K nearest neighbors
		nearest_neighbors = distances[0:K]
		#tally votes
		votes = np.zeros(10, dtype=np.int8)
		for i in range(K):
			votes[sample_labels[nearest_neighbors[i][1]]] = votes[sample_labels[nearest_neighbors[i][1]]] + 1
		#Pick vote majority
		decision = np.argmax(votes)
		labels.append(decision)
	return labels
'''
args: 
sample - tuple of training sample and corresponding labels
test - test_data and labels
We project sample into eigenspace and do k Nearest Neighbors
'''
def classify(sample, test, mean, eigen_mat, K):
	sample_projections = []
	train_images = sample[0]
	train_images = vectorize_images(train_images)
	sample_projections = project_vectors(train_images, mean, eigen_mat)
	test_vectors = vectorize_images(test[0])
	test_projections = project_vectors(test_vectors, mean, eigen_mat)
	labels = kNN(K, sample_projections, sample[1], test_projections)
	test_labels = test[1]
	success = 0
	fail = 0
	test_labels = test_labels.reshape((np.shape(test_labels)[1]))
	for i in range(len(test_labels)):
		if test_labels[i] == labels[i]:
			success = success + 1
		else:
			fail = fail + 1
	return float(success)/float(success + fail)

def main():
	#10000 28 x 28 test images
	testImages = load("testImages.bin", (28, 28, 10000))
	#10000 labels
	testLabels = load("testLabels.bin", (1, 10000))
	#60000 28 x 28  images
	trainImages = load("trainImages.bin", (28, 28, 60000))
	# 60000 labels
	trainLabels = load("trainLabels.bin", (1, 60000))

	sample = sample_images(trainImages, trainLabels, 200)
	images = sample[0]
	labels = sample[1]
	images = vectorize_images(images)
	eigen_info = hw1FindEigendigits(images)
	mean = eigen_info[0]
	eigen_mat = eigen_info[1]
	#Display Eigenvectors
	for i in np.random.randint(np.shape(eigen_mat)[1], size=5):
		eigenimg = np.reshape(eigen_mat[:,i], (28, 28, 1))
		#plt.imshow(eigenimg[:, :, 0])
		#plt.show()

	testim = sample_images(testImages, testLabels, 5)[0]
	testim = vectorize_images(testim)
	for i in np.random.randint(np.shape(testim)[1], size=5):
		projection = project(testim[:,i], mean, eigen_mat)
		img = np.reshape(testim[:,i], (28, 28, 1))
		#plt.imshow(img[:, :, 0], cmap='gray')
		#plt.show()		
		unprojection = unproject(projection, mean, eigen_mat)
		unprojection = np.reshape(unprojection, (28, 28, 1))
		#plt.imshow(unprojection[:, :, 0], cmap='gray')
		#plt.show()
	easy =  classify(sample, (testImages[:,:,5000:10000], testLabels[:,5000:]), mean, eigen_mat, 5)
	hard = classify(sample, (testImages[:,:,0:5000], testLabels[:,0:5000]), mean, eigen_mat, 5)
	print str(easy) + " " + str(hard)

if __name__ == '__main__':
	main()