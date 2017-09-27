
import numpy as np
from bowfeature import BOWFeature

class Perceptron:
	def __init__(self, feature, classes):

		self.feature_function = feature #BOWFeature(train_x, train_y)
		self.classes = classes #list(set(train_y))


	def train(self, train_x, train_y):
		print("Start Training:")

		k = self.feature_function.size()
		# Initialize weights to zeros
		self.weights = np.array([ ([0]*(k-1))+[1] for _ in xrange(0, len(self.classes))])
		self.weights = np.asfarray(self.weights)
		weights_sum = self.weights.copy()

		for i in xrange(0, len(train_y)):
			if(i % 300 == 0):
				print("{0:.2f}%".format(100 * (i/float(len(train_y)))))

			example = train_x[i]
			#X = self.get_bow_feature(tokens)
			X = self.feature_function.get_feature(example)
			#X = np.clip(X, 0, 1)

			y = self.classes.index(train_y[i])

			# Predict using current weights
			plabel = self._predict(X)
			py = self.classes.index(plabel)

			# Check if we got the correct answer
			if(y != py):
				# Update weights
				self.weights[y] += X
				self.weights[py] -= X
				weights_sum += self.weights

			else:
				# Nothing to do
				pass

		self.weights = weights_sum * 1.0/len(train_y)

		print("Training is Done")


	def predict(self, example):
		X = self.feature_function.get_feature(example)
		#X = np.clip(X, 0, 1)

		return self._predict(X)


	def _predict(self, X):
		scores = [0] * len(self.classes)
		for j in xrange(0, len(self.classes)):
			W = self.weights[j]
			score = np.dot(W, X)
			scores[j] = score
		py = np.argmax(scores)

		return self.classes[py]


