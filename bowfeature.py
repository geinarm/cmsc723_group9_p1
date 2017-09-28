
import numpy as np

class BOWFeature():
	def __init__(self, train_x, train_y, binary=False):

		self.classes = list(set(train_y))
		self.vocab = list(set([item for l_ in train_x for item in l_]))
		self.K = len(self.vocab)
		self.binary = binary

		self.vocab_dict = {}
		for i in xrange(0, self.K):
			w = self.vocab[i]
			self.vocab_dict[w] = i


	def size(self):
		return len(self.vocab) + 1 		# plus one for the bias

	def get_feature(self, example):
		X = ([0] * self.K)+[1]

		for w in example:
			if(w in self.vocab_dict):
				i = self.vocab_dict[w]
				X[i] += 1

		if(self.binary):
			X = np.clip(X, 0, 1)

		return np.array(X)


	## Returns a matrix where the i-th row is the feature vector 
	## corresponding to the i-th example in the list
	def get_features(self, examples):
		M = []

		for e in examples:
			x = self.get_feature(e)
			M.append(x)

		return np.array(M)

