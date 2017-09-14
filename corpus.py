
import numpy as np

class Corpus:
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
		self.matrix = count_dict.values()


	def get_bow_vector(self, tokens):

		X = [0] * len(self.vocabulary)

		for i in xrange(len(self.vocabulary)):
			w = self.vocabulary[i]

			X[i] = tokens.count(w)

		return np.array(X)


	def get_weights(self, label):
		M = np.array(self.matrix)
		class_index = self.classes.index(label)

		prior = np.divide(self.class_count, float(self.N))

		c = M[:, class_index]
		t = np.sum(M, axis=1)
		t = np.asfarray(t)
		W = np.divide(c, t)
		W = np.append(W, prior[class_index])

		return W


			

