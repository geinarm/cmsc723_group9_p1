
import numpy as np

class BOWFeature():
	def __init__(self, train_x, train_y):

		self.classes = list(set(train_y))
		self.vocab = list(set([item for l_ in train_x for item in l_]))
		self.num_classes = len(self.classes)
		self.N = len(train_y)
		self.K = len(self.vocab)

		self.vocab_dict = {}
		for i in xrange(0, self.K):
			w = self.vocab[i]
			self.vocab_dict[w] = i


	def get_bow_feature(self, tokens):
		X = ([0] * self.K)+[1]

		for w in tokens:
			if(w in self.vocab_dict):
				i = self.vocab_dict[w]
				X[i] = 1

		return np.array(X)


	## Returns a matrix where the i-th row is the feature vector 
	## corresponding to the i-th example in the list
	def get_bow_features(self, Examples):
		M = []

		for e in Examples:
			x = self.get_bow_feature(e)
			M.append(x)

		return np.array(M)

