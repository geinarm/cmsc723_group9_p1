
import numpy as np

class Perceptron:
	def __init__(self, train_x, train_y):

		
		self.classes = list(set(train_y))
		self.num_classes = len(self.classes)
		self.class_count = [0]*self.num_classes
		self.N = len(train_y)

		count_dict = {}
		for i in xrange(0, len(train_y)):
			y = train_y[i]
			X = train_x[i]
			class_index = self.classes.index(y)
			self.class_count[class_index] += 1

			for w in X:
				if w not in count_dict:
					count_dict[w] = [0]*self.num_classes
				
				count_dict[w][class_index] += 1

		self.vocabulary = count_dict.keys()
		self.matrix = np.array(count_dict.values())  		# Occurence count matrix (|V| * 6)


	def get_bow_feature(self, tokens):
		X = ([0] * len(self.vocabulary))+[1]

		for i in xrange(len(self.vocabulary)):
			w = self.vocabulary[i]

			if(w in tokens):
				X[i] = 1
			#X[i] = tokens.count(w)

		return np.array(X)


	def train(self, train_x, train_y):
		print("Start Training:")
		# Initialize weights to zeros
		self.weights = np.array([ ([0]*len(self.vocabulary))+[1] for _ in xrange(0, self.num_classes)])
		self.weights = np.asfarray(self.weights)
		weights_sum = self.weights.copy()

		for i in xrange(0, len(train_y)):
			if(i % 300 == 0):
				print("{0:.2f}%".format(100 * (i/float(len(train_y)))))

			tokens = train_x[i]
			X = self.get_bow_feature(tokens)
			y = self.classes.index(train_y[i])

			# Predict using current weights
			plabel = self.predict(X)
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

		self.weights = weights_sum * 1.0/len(self.vocabulary)

		print("Training is Done")


	def predict(self, X):
		scores = [0]*self.num_classes
		for j in xrange(0, self.num_classes):
			W = self.weights[j]
			score = np.dot(W, X)
			scores[j] = score
		py = np.argmax(scores)

		return self.classes[py]

