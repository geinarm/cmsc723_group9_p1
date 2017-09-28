
import numpy as np
import math
from bowfeature import BOWFeature

class NaiveBayes:
	def __init__(self, feature, classes):

		self.feature_function = feature #BOWFeature(train_x, train_y)
		self.classes = classes #list(set(train_y))


	def train(self, train_x, train_y):
		print("Start Training:")

		## Count feature(words) occurences
		n = len(train_x)
		k = self.feature_function.size()
		count_matrix = np.array([ ([0]*len(self.classes)) for _ in xrange(0, k)])
		count_matrix = np.asfarray(count_matrix)
		class_count = [0] * len(self.classes)

		for i in xrange(0, n):
			if(i % 300 == 0):
				print("{0:.2f}%".format(100 * (i/float(len(train_y)))))

			example = train_x[i]
			label = train_y[i]

			x = self.feature_function.get_feature(example)
			y = self.classes.index(label)
			class_count[y] += 1

			count_matrix[:, y] += x
			

		## Q2 - 1
		print(self.classes)
		#print(class_count)
		print("C(Si | time) = {0}".format(count_matrix[994, :]))
		print("C(s_i | loss) = {0}".format(count_matrix[7596, :]))
		print("C(s_i | export) = {0}".format(count_matrix[159, :]))
		count_matrix = np.delete(count_matrix, k-1, 0)	# remove count of bias element

		## Calculate probability
		prior = np.array(class_count) / float(n)

		count_matrix = count_matrix + 1
		col_sum = np.sum(count_matrix, axis=0)
		W = count_matrix / col_sum
		W = np.vstack([W, prior])

		self.weights = W

		## Q2 - 2
		#print(self.classes)
		#print(prior)
		#print("P(Si | time) = {0}".format(self.weights[994, :]))
		#print("P(s_i | loss) = {0}".format(self.weights[7596, :]))
		#rint("P(s_i | export) = {0}".format(self.weights[159, :]))

		self.weights = np.log(self.weights) 	# Log Propability


	def classify(self, example):
		## Hack
		if self.feature_function.__class__.__name__ == 'BOWFeature':
			self.feature_function.binary = True
		elif self.feature_function.__class__.__name__ == 'LeskFeature':
			self.feature_function.binary = True

		X = self.feature_function.get_feature(example)
		X = np.clip(X, 0, 1)  ## binary feature. Ignore count

		score = [0] * len(self.classes)
		for i in xrange(0, len(self.classes)):
			W = self.weights[:, i]
			score[i] = np.dot(X, W)

		##Q2 - 3
		#print(self.classes)
		#print(score)

		pi = np.argmax(score)
		return self.classes[pi]



			

