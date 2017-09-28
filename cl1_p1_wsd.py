"""
CMSC723 / INST725 / LING723 -- Fall 2016
Project 1: Implementing Word Sense Disambiguation Systems
"""
import numpy as np
from bowfeature import BOWFeature
from pcafeature import PCAFeature
from leskfeature import LeskFeature

class Perceptron:
	def __init__(self, feature, classes):

		self.feature_function = feature #BOWFeature(train_x, train_y)
		self.classes = classes #list(set(train_y))

		## Hack
		if self.feature_function.__class__.__name__ == 'BOWFeature':
			self.feature_function.binary = True
		#elif self.feature_function.__class__.__name__ == 'LeskFeature':
		#	self.feature_function.binary = True


	def train(self, train_x, train_y):
		print("Start Training:")

		k = self.feature_function.size()
		# Initialize weights to zeros
		self.weights = np.array([ ([0]*(k-1))+[1] for _ in xrange(0, len(self.classes))])
		self.weights = np.asfarray(self.weights)
		weights_sum = self.weights.copy()

		itt = 0

		for j in xrange(0, 3): 		## Itterations
			print("Itteration {0}".format(j+1))
			for i in xrange(0, len(train_y)):
				if(i % 300 == 0):
					print("{0:.2f}%".format(100 * (i/float(len(train_y)))))

				example = train_x[i]
				X = self.feature_function.get_feature(example)
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
					itt += 1

		self.weights = weights_sum * 1.0/itt

		print("Training is Done")


	def predict(self, example):
		X = self.feature_function.get_feature(example)

		return self._predict(X)


	def _predict(self, X):
		scores = [0] * len(self.classes)
		for j in xrange(0, len(self.classes)):
			W = self.weights[j]
			score = np.dot(W, X)
			scores[j] = score
		py = np.argmax(scores)

		return self.classes[py]
##Perceptron






"""
 read one of train, dev, test subsets 
 
 subset - one of train, dev, test
 
 output is a tuple of three lists
	labels: one of the 6 possible senses <cord, division, formation, phone, product, text >
	targets: the index within the text of the token to be disambiguated
	texts: a list of tokenized and normalized text input (note that there can be multiple sentences)

"""
import nltk 
def read_dataset(subset):
	labels = []
	texts = []
	targets = []
	if subset in ['train', 'dev', 'test']:
		with open('data/wsd_'+subset+'.txt') as inp_hndl:
			for example in inp_hndl:
				label, text = example.strip().split('\t')
				text = nltk.word_tokenize(text.lower().replace('" ','"'))
				if 'line' in text:
					ambig_ix = text.index('line')
				elif 'lines' in text:
					ambig_ix = text.index('lines')
				else:
					ldjal
				targets.append(ambig_ix)
				labels.append(label)
				texts.append(text)
		return (labels, targets, texts)
	else:
		print '>>>> invalid input !!! <<<<<'

"""
computes f1-score of the classification accuracy

gold_labels - is a list of the gold labels
predicted_labels - is a list of the predicted labels

output is a tuple of the micro averaged score and the macro averaged score

"""
import sklearn.metrics
def eval(gold_labels, predicted_labels):
	return ( sklearn.metrics.f1_score(gold_labels, predicted_labels, average='micro'),
			 sklearn.metrics.f1_score(gold_labels, predicted_labels, average='macro') )


"""
a helper method that takes a list of predictions and writes them to a file (1 prediction per line)
predictions - list of predictions (strings)
file_name - name of the output file
"""
def write_predictions(predictions, file_name):
	with open(file_name, 'w') as outh:
		for p in predictions:
			outh.write(p+'\n')

"""
Trains a naive bayes model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.
"""
from naiveBayes import NaiveBayes
import numpy as np
def run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):


	classes = list(set(train_labels))

	#feature = LeskFeature(classes)
	feature = BOWFeature(train_texts, train_labels)
	#feature = PCAFeature(train_texts, train_labels, 500)

	NB = NaiveBayes(feature, classes)
	NB.train(train_texts, train_labels)

	gold = []
	pred = []
	for i in xrange(0, len(test_texts)):
		if(i % 100 == 0):
			print(i)

		c = NB.classify(test_texts[i])
		pred.append(c)
		gold.append(test_labels[i])

	return eval(gold, pred)

"""
Trains a perceptron model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
import numpy as np
def run_bow_perceptron_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	
	classes = list(set(train_labels))

	feature = BOWFeature(train_texts, train_labels)
	percept = Perceptron(feature, classes)
	percept.train(train_texts, train_labels)

	gold = []
	pred = []
	print("Classify test set:")
	for i in xrange(0, len(test_texts)):
		if(i % 300 == 0):
			print("{0:.2f}%".format(100 * (i/float(len(test_texts)))))

		c = percept.predict(test_texts[i])
		pred.append(c)
		gold.append(test_labels[i])
	print("Classification Done")

	write_predictions(pred, 'q3p3.txt')

	return eval(gold, pred)


"""
Trains a naive bayes model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_naivebayes_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):

	classes = list(set(train_labels))

	feature = LeskFeature(classes)

	NB = NaiveBayes(feature, classes)
	NB.train(train_texts, train_labels)

	gold = []
	pred = []
	for i in xrange(0, len(test_texts)):
		if(i % 100 == 0):
			print(i)

		c = NB.classify(test_texts[i])
		pred.append(c)
		gold.append(test_labels[i])

	return eval(gold, pred)

"""
Trains a perceptron model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_perceptron_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):

	classes = list(set(train_labels))

	feature = PCAFeature(train_texts, train_labels, 500)

	percept = Perceptron(feature, classes)
	percept.train(train_texts, train_labels)

	gold = []
	pred = []
	print("Classify test set:")
	for i in xrange(0, len(test_texts)):
		if(i % 300 == 0):
			print("{0:.2f}%".format(100 * (i/float(len(test_texts)))))

		c = percept.predict(test_texts[i])
		pred.append(c)
		gold.append(test_labels[i])
	print("Classification Done")

	return eval(gold, pred)


if __name__ == "__main__":
	# reading, tokenizing, and normalizing data
	train_labels, train_targets, train_texts = read_dataset('train')
	dev_labels, dev_targets, dev_texts = read_dataset('dev')
	test_labels, test_targets, test_texts = read_dataset('test')

	## running the classifier

	#test_scores = run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
	#           dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

	test_scores = run_bow_perceptron_classifier(train_texts, train_targets, train_labels, 
			dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

	print test_scores
