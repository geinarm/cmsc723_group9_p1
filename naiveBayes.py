
import numpy as np
import math

class NaiveBayes:
	def __init__(self, labels, tokens):
		self.classes = list(set(labels))
		self.num_classes = len(self.classes)
		self.class_count = [0]*self.num_classes
		self.N = len(labels)

		count_dict = {}
		for i in xrange(0, len(labels)):
			label = labels[i]
			token = tokens[i]
			class_index = self.classes.index(label)
			self.class_count[class_index] += 1

			for w in token:
				if w not in count_dict:
					count_dict[w] = [0]*self.num_classes
				
				count_dict[w][class_index] += 1

		self.vocabulary = count_dict.keys()
		self.matrix = np.array(count_dict.values())  		# Occurence count matrix (|V| * 6)

		self.weights = []
		for i in xrange(0, self.num_classes):
			W = self.calculate_weights(self.classes[i])
			self.weights.append(W)


	def calculate_weights(self, label):
		class_index = self.classes.index(label)

		prior = self.class_count[class_index] / float(self.N)

		c = self.matrix[:, class_index] 	# Occurence count in class for each word. Large column vector
		c = np.add(c, 1)					# Laplace Smoothing
		c_sum = np.sum(c) 					# Total word occurances in class

		W = np.divide(c, float(c_sum))		# Propability
		W = np.log(W) 						# Log Propability
		W = np.append(W, math.log(prior)) 	# Add the prior to the end of the weight vector

		return W


	def get_bow_vector(self, tokens):

		X = [0] * len(self.vocabulary)

		for i in xrange(len(self.vocabulary)):
			w = self.vocabulary[i]

			#X[i] = tokens.count(w)
			if(w in tokens):
				X[i] = 1

		return np.array(X)


	def classify(self, tokens):
		X = self.get_bow_vector(tokens)
		X = np.append(X, 1)

		score = [0]*self.num_classes
		for i in xrange(0, len(self.classes)):
			W = self.weights[i]
			score[i] = np.dot(X, W)

		pi = np.argmax(score)
		return self.classes[pi]



			

