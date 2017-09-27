
import numpy as np
import time
from bowfeature import BOWFeature

class PCAFeature():
	def __init__(self, train_x, train_y, k):
		
		self.k = k
		self.bow = BOWFeature(train_x, train_y)

		X = self.bow.get_features(train_x)
		X = np.asfarray(X)
		
		## Subtract the mean to center data at the origin
		Mu = np.mean(X, 0)
		X = X - Mu

		## Singular Value Decompisition
		print('Do SVD')
		U, s, V = np.linalg.svd(X, full_matrices=False)
		print('SVD Done')

		## Save eignen vectors. Used later to project into new feature space
		self.V = V


	def size(self):
		return self.k+1  # plus one for the bias


	def get_feature(self, ex):

		x = self.bow.get_feature(ex, binary=True)
		x_ = np.dot(self.V, x)
		x_ = np.dot(self.V, x.transpose()).transpose()

		x_ = x_[0:self.k]
		x_ = np.append(x_, [1])

		return x_


	def get_features(self, examples):
		n = len(examples)

		X = self.bow.get_features(examples, binary=True)
		X_ = np.dot(self.V, X)
		X_ = np.dot(self.V, X.transpose()).transpose()

		X_ = X_[:, 0:self.k]
		X_ = np.append(X_, [[1]*n], axis=1)

		return X_


import cl1_p1_wsd as cl1

if __name__ == "__main__":
	#train_labels, train_targets, train_texts = cl1.read_dataset('train')
	#pca = PCAFeature(train_texts, train_labels, 200)
	#x_ = pca.get_feature(M[0, :])
	#print(x_)

	X = np.array([ \
	   [0.0898708,   0.0855400,  -0.0944973], \
	   [0.0359912,  -0.0914864,  -0.0288341], \
	  [-0.2564242,  -0.0132445,   0.2735117], \
	   [0.2698768,   0.0368777,  -0.3769418], \
	   [0.3360477,   0.0108863,  -0.3370864], \
	  [-0.0024894,   0.1590193,   0.0075606], \
	  [-0.1678390,  -0.0171557,   0.2565151], \
	  [-0.0811866,   0.1609380,   0.0499579], \
	   [0.0140084,  -0.1769449,  -0.0281955], \
	   [0.1533342,  -0.1646840,  -0.1567181], \
	  [-0.0353900,   0.1374947,  -0.0185367], \
	  [-0.1155516,   0.0719622,   0.1233399], \
	   [0.3622019,  -0.0061652,  -0.4158134], \
	   [0.1517388,  -0.0482249,  -0.1796406], \
	   [0.2002729,  -0.1456160,  -0.2222606], \
	  [-0.1110595,   0.1093557,   0.1261832], \
	  [-0.4876974,  -0.0638777,   0.5463041], \
	  [-0.2247021,   0.1764028,   0.2369173], \
	   [0.3008229,  -0.1200166,  -0.3417893], \
	  [-0.0396668,   0.0873703,   0.0746332], \
	  [-0.2530123,   0.0529436,   0.3339835], \
	  [-0.0731627,   0.1624831,   0.1226515], \
	  [-0.2214905,  -0.1561934,   0.1688039], \
	  [-0.3664830,  -0.0043208,   0.4318502], \
	   [0.1374741,  -0.0811718,  -0.1504727], \
	   [0.4299418,   0.0451998,  -0.5129476], \
	   [0.1790183,  -0.1462799,  -0.2243107], \
	   [0.2257256,   0.0033128,  -0.2264165], \
	  [-0.4237678,  -0.0412640,   0.5335044], \
	  [-0.0264023,  -0.0231403,   0.0287448], \
		])


	k = 2
	
	## Subtract the mean to center data at the origin
	Mu = np.mean(X, 0)
	#X = X - Mu

	## Singular Value Decompisition
	U, s, V = np.linalg.svd(X, full_matrices=False)

	X_ = np.dot(V, X.transpose()).transpose()

	print(X)
	print(V)
	print(X_)


