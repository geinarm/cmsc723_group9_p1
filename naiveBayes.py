
import numpy as np
import math
from bowfeature import BOWFeature

class NaiveBayes:
	def __init__(self, train_x, train_y):

		self.feature_function = BOWFeature(train_x, train_y)
		self.vocabulary = self.feature_function.vocab

		self.classes = list(set(train_y))


	def train(self, train_x, train_y):
		print("Start Training:")

		## Count feature(words) occurences
		n = len(train_x)
		k = self.feature_function.size() #len(self.vocabulary)
		count_matrix = np.array([ ([0]*len(self.classes)) for _ in xrange(0, k)])
		class_count = [0] * len(self.classes)
		print(count_matrix.shape)

		for i in xrange(0, n):
			if(i % 300 == 0):
				print("{0:.2f}%".format(100 * (i/float(len(train_y)))))

			example = train_x[i]
			label = train_y[i]

			x = self.feature_function.get_feature(example)
			y = self.classes.index(label)
			class_count[y] += 1

			count_matrix[:, y] += x
			
		count_matrix = np.delete(count_matrix, k-1, 0)	# remove count of bias element
		count_matrix = np.asfarray(count_matrix)		# convert to float just to be sure

		print('Sum2 :')
		print(np.sum(count_matrix))

		## Calculate probability
		self.weights = []
		for i in xrange(0, len(self.classes)):
			prior = class_count[i] / float(n)

			c = count_matrix[:, i] 				# Occurence count in class i for each word. Large column vector
			c = np.add(c, 1)					# Laplace Smoothing
			c_sum = np.sum(c) 					# Total word occurances in class

			W = np.divide(c, float(c_sum))		# Propability
			W = np.log(W) 						# Log Propability
			W = np.append(W, math.log(prior)) 	# Add the prior to the end of the weight vector

			self.weights.append(W)


	def classify(self, example):
		## Hack
		if self.feature_function.__class__.__name__ == 'BOWFeature':
			self.feature_function.binary = True

		#X = self.get_bow_vector(tokens)
		X = self.feature_function.get_feature(example)
		X = np.clip(X, 0, 1)  ## binary feature. Ignore count
		#X = np.append(X, 1)

		score = [0] * len(self.classes)
		for i in xrange(0, len(self.classes)):
			W = self.weights[i]
			score[i] = np.dot(X, W)

		pi = np.argmax(score)
		return self.classes[pi]



			

