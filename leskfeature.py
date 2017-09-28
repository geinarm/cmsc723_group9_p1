
import numpy as np
import nltk 
from nltk.tokenize import RegexpTokenizer

class LeskFeature():
	def __init__(self, classes):
		self.classes = classes
		self.labels = []
		self.texts = []
		self.binary = False


		tokenizer = RegexpTokenizer(r'\w+')
		with open('data/definition.txt') as f:
			for line in f:
				label, text = line.strip().split('\t')
				text = text.lower().replace('" ','"')
				text = tokenizer.tokenize(text)

				self.labels.append(label)
				self.texts.append(text)

		self.vocab = list(set([item for l_ in self.texts for item in l_]))

		self.vocab_dict = {}
		for i in xrange(0, len(self.vocab)):
			w = self.vocab[i]
			self.vocab_dict[w] = i

		"""
		count_matrix = np.array([ ([0]*len(self.labels)) for _ in self.vocab ])

		for i in xrange(0, len(self.vocab)):
			word = self.vocab[i]

			for c in self.classes:
				class_index = self.labels.index(c)
				defintion = self.texts[class_index]

				count = defintion.count(word)
				count_matrix[i, class_index] = count

		count_matrix = np.asfarray(count_matrix)
		row_sum = np.sum(count_matrix, axis=1)
		self.word_weights = (count_matrix.transpose() / row_sum).transpose()
		"""


	def size(self):
		return len(self.vocab) + 1 		# plus one for the bias


	def get_feature(self, example):
		X = ([0] * len(self.vocab))+[1]

		for w in example:
			if(w in self.vocab_dict):
				i = self.vocab_dict[w]
				X[i] += 1

		if(self.binary):
			X = np.clip(X, 0, 1)

		return np.array(X)

