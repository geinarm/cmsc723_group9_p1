"""
CMSC723 / INST725 / LING723 -- Fall 2016
Project 1: Implementing Word Sense Disambiguation Systems
"""

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

		## Q2 - 1
		#print("time: {0}".format(self.vocab.index('time')))
		#print("loss: {0}".format(self.vocab.index('loss')))
		#print("export: {0}".format(self.vocab.index('export')))


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
##BOWFeature

class PCAFeature():
	def __init__(self, train_x, train_y, k):
		
		self.k = k
		self.bow = BOWFeature(train_x, train_y)

		X = self.bow.get_features(train_x)
		X = np.asfarray(X)
		
		## Subtract the mean to center data at the origin
		Mu = np.mean(X, 0)
		X = X - Mu

		## Singular Value Decompisition
		#print('Do SVD')
		U, s, V = np.linalg.svd(X, full_matrices=False)
		#print('SVD Done')

		## Save eignen vectors. Used later to project into new feature space
		self.V = V


	def size(self):
		return self.k+1  # plus one for the bias


	def get_feature(self, ex):

		x = self.bow.get_feature(ex)
		x_ = np.dot(self.V, x)
		#x_ = np.dot(self.V, x.transpose()).transpose()

		x_ = x_[0:self.k]
		x_ = np.append(x_, [1])

		return x_


	def get_features(self, examples):
		n = len(examples)

		X = self.bow.get_features(examples)
		X_ = np.dot(self.V, X)
		X_ = np.dot(self.V, X.transpose()).transpose()

		X_ = X_[:, 0:self.k]
		X_ = np.append(X_, [[1]*n], axis=1)

		return X_
##PCAFeature


from nltk.tokenize import RegexpTokenizer
class LeskFeature():
	def __init__(self, classes):
		self.classes = classes
		self.binary = False

		## Moved from data to hardcoded :/
		self.labels = ['cord', 'division', 'formation', 'phone', 'product', 'text']
		self.texts = [
  				['long', 'thin', 'flexible', 'string', 'or', 'rope', 'made', 'from', 'several', 'twisted', 'strands', 'hang', 'the', 'picture', 'from', 'a', 'rail', 'on', 'a', 'length', 'of', 'cord', 'string', 'thread', 'thong', 'lace', 'ribbon', 'strap', 'tape', 'tie', 'line', 'rope', 'cable', 'wire', 'ligature', 'morea', 'length', 'of', 'string', 'or', 'rope', 'used', 'to', 'fasten', 'or', 'move', 'a', 'specified', 'object', 'plural', 'noun', 'cords', 'a', 'dressing', 'gown', 'cord', 'an', 'anatomical', 'structure', 'resembling', 'a', 'length', 'of', 'cord', 'e', 'g', 'the', 'spinal', 'cord', 'the', 'umbilical', 'cord', 'the', 'baby', 'was', 'still', 'attached', 'to', 'its', 'mother', 'by', 'the', 'cord', 'a', 'flexible', 'insulated', 'cable', 'used', 'for', 'carrying', 'electric', 'current', 'to', 'an', 'appliance'], \
  				['the', 'action', 'of', 'separating', 'something', 'into', 'parts', 'or', 'the', 'process', 'of', 'being', 'separated', 'the', 'division', 'of', 'the', 'land', 'into', 'small', 'fields', 'dividing', 'up', 'breaking', 'up', 'breakup', 'carving', 'up', 'splitting', 'dissection', 'bisection', 'more', 'the', 'distribution', 'of', 'something', 'separated', 'into', 'parts', 'the', 'division', 'of', 'his', 'estates', 'between', 'the', 'two', 'branches', 'of', 'his', 'family', 'sharing', 'out', 'dividing', 'up', 'parceling', 'out', 'dishing', 'out', 'allocation', 'allotment', 'apportionment', 'more', 'an', 'instance', 'of', 'members', 'of', 'a', 'legislative', 'body', 'separating', 'into', 'two', 'groups', 'to', 'vote', 'for', 'or', 'against', 'a', 'bill', 'plural', 'noun', 'divisions', 'the', 'new', 'clause', 'was', 'agreed', 'without', 'a', 'division', 'the', 'action', 'of', 'splitting', 'the', 'roots', 'of', 'a', 'perennial', 'plant', 'into', 'parts', 'to', 'be', 'replanted', 'separately', 'as', 'a', 'means', 'of', 'propagation', 'the', 'plant', 'can', 'also', 'be', 'easily', 'increased', 'by', 'division', 'in', 'autumn', 'the', 'action', 'of', 'dividing', 'a', 'wider', 'class', 'into', 'two', 'or', 'more', 'subclasses'],  \
  				['the', 'action', 'of', 'forming', 'or', 'process', 'of', 'being', 'formed', 'the', 'formation', 'of', 'the', 'great', 'rift', 'valley', 'emergence', 'coming', 'into', 'being', 'genesis', 'development', 'evolution', 'shaping', 'origination', 'the', 'formation', 'of', 'the', 'island', 's', 'sand', 'ridges', 'establishment', 'setting', 'up', 'start', 'initiation', 'institution', 'foundation', 'inception', 'creation', 'inauguration', 'launch', 'flotation', 'the', 'formation', 'of', 'a', 'new', 'government', 'antonyms', 'destruction', 'disappearance', 'dissolution', 'a', 'structure', 'or', 'arrangement', 'of', 'something', 'a', 'cloud', 'formation', 'configuration', 'arrangement', 'pattern', 'array', 'alignment', 'positioning', 'disposition', 'order', 'the', 'aircraft', 'were', 'flying', 'in', 'tight', 'formation', 'a', 'formal', 'arrangement', 'of', 'aircraft', 'in', 'flight', 'or', 'troops', 'a', 'battle', 'formation', 'an', 'assemblage', 'of', 'rocks', 'or', 'series', 'of', 'strata', 'having', 'some', 'common', 'characteristic'], \
  				['a', 'telephone', 'a', 'few', 'seconds', 'later', 'the', 'phone', 'rang', 'telephone', 'cell', 'phone', 'cell', 'car', 'phone', 'cordless', 'phone', 'speakerphone', 'extension', 'informalblower', 'horn', 'keitai', 'she', 'tried', 'to', 'reach', 'you', 'on', 'your', 'phone', 'headphones', 'or', 'earphones', 'verb', 'phone', '3rd', 'person', 'present', 'phones', 'past', 'tense', 'phoned', 'past', 'participle', 'phoned', 'gerund', 'or', 'present', 'participle', 'phoning', 'call', 'someone', 'on', 'the', 'telephone', 'he', 'phoned', 'her', 'at', 'work', 'telephone', 'call', 'give', 'someone', 'a', 'call', 'informalcall', 'up', 'give', 'someone', 'a', 'buzz', 'get', 'someone', 'on', 'the', 'horn', 'blower', 'i', 'll', 'phone', 'you', 'later'], \
  				['an', 'article', 'or', 'substance', 'that', 'is', 'manufactured', 'or', 'refined', 'for', 'sale', 'marketing', 'products', 'and', 'services', 'a', 'substance', 'produced', 'during', 'a', 'natural', 'chemical', 'or', 'manufacturing', 'process', 'waste', 'products', 'artifact', 'commodity', 'manufactured', 'article', 'creation', 'invention', 'goods', 'wares', 'merchandise', 'produce', 'a', 'household', 'product', 'a', 'thing', 'or', 'person', 'that', 'is', 'the', 'result', 'of', 'an', 'action', 'or', 'process', 'his', 'daughter', 'the', 'product', 'of', 'his', 'first', 'marriage', 'result', 'consequence', 'outcome', 'effect', 'upshot', 'fruit', 'by', 'product', 'spin', 'off', 'his', 'skill', 'is', 'a', 'product', 'of', 'experience', 'a', 'person', 'whose', 'character', 'and', 'identity', 'have', 'been', 'formed', 'by', 'a', 'particular', 'period', 'or', 'situation', 'an', 'aging', 'academic', 'who', 'is', 'a', 'product', 'of', 'the', '1960s', 'commercially', 'manufactured', 'articles', 'especially', 'recordings', 'viewed', 'collectively', 'too', 'much', 'product', 'of', 'too', 'little', 'quality'], \
  				['a', 'book', 'or', 'other', 'written', 'or', 'printed', 'work', 'regarded', 'in', 'terms', 'of', 'its', 'content', 'rather', 'than', 'its', 'physical', 'form', 'a', 'text', 'that', 'explores', 'pain', 'and', 'grief', 'book', 'work', 'written', 'work', 'printed', 'work', 'document', 'a', 'text', 'that', 'explores', 'pain', 'and', 'grief', 'a', 'piece', 'of', 'written', 'or', 'printed', 'material', 'regarded', 'as', 'conveying', 'the', 'authentic', 'or', 'primary', 'form', 'of', 'a', 'particular', 'work', 'in', 'some', 'passages', 'it', 'is', 'difficult', 'to', 'establish', 'the', 'original', 'text', 'written', 'or', 'printed', 'words', 'typically', 'forming', 'a', 'connected', 'piece', 'of', 'work', 'stylistic', 'features', 'of', 'journalistic', 'text', 'data', 'in', 'the', 'form', 'of', 'words', 'or', 'alphabetic', 'characters', 'the', 'main', 'body', 'of', 'a', 'book', 'or', 'other', 'piece', 'of', 'writing', 'as', 'distinct', 'from', 'other', 'material', 'such', 'as', 'notes', 'appendices', 'and', 'illustrations', 'the', 'pictures', 'are', 'clear', 'and', 'relate', 'well', 'to', 'the', 'text', 'words', 'wording', 'writing', 'a', 'script', 'or', 'libretto', 'a', 'written', 'work', 'chosen', 'or', 'assigned', 'as', 'a', 'subject', 'of', 'study', 'the', 'book', 'is', 'intended', 'as', 'a', 'secondary', 'text', 'for', 'religion', 'courses', 'a', 'textbook', 'textbook', 'book', 'material', 'academic', 'texts', 'a', 'passage', 'from', 'the', 'bible', 'or', 'other', 'religious', 'work', 'especially', 'when', 'used', 'as', 'the', 'subject', 'of', 'a', 'sermon', 'passage', 'extract', 'excerpt', 'quotation', 'verse', 'line', 'reading', 'a', 'text', 'from', 'the', 'first', 'book', 'of', 'samuel', 'a', 'subject', 'or', 'theme', 'for', 'a', 'discussion', 'or', 'exposition', 'he', 'took', 'as', 'his', 'text', 'the', 'fact', 'that', 'australia', 'is', 'paradise', 'a', 'text', 'message', 'fine', 'large', 'handwriting', 'used', 'especially', 'for', 'manuscripts', 'verb', 'text', '3rd', 'person', 'present', 'texts', 'past', 'tense', 'texted', 'past', 'participle', 'texted', 'gerund', 'or', 'present', 'participle', 'texting', 'send', 'a', 'text', 'message', 'to', 'i', 'thought', 'it', 'was', 'fantastic', 'that', 'he', 'took', 'the', 'trouble', 'to', 'text', 'me'] \
			]

		"""
		tokenizer = RegexpTokenizer(r'\w+')
		with open('data/definition.txt') as f:
			for line in f:
				label, text = line.strip().split('\t')
				text = text.lower().replace('" ','"')
				text = tokenizer.tokenize(text)

				self.labels.append(label)
				self.texts.append(text)
		"""

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
##LeskFeature


class NaiveBayes:
	def __init__(self, feature, classes):

		self.feature_function = feature #BOWFeature(train_x, train_y)
		self.classes = classes #list(set(train_y))


	def train(self, train_x, train_y):
		#print("Start Training:")

		## Count feature(words) occurences
		n = len(train_x)
		k = self.feature_function.size()
		count_matrix = np.array([ ([0]*len(self.classes)) for _ in xrange(0, k)])
		count_matrix = np.asfarray(count_matrix)
		class_count = [0] * len(self.classes)

		for i in xrange(0, n):
			#if(i % 300 == 0):
			#	print("{0:.2f}%".format(100 * (i/float(len(train_y)))))

			example = train_x[i]
			label = train_y[i]

			x = self.feature_function.get_feature(example)
			y = self.classes.index(label)
			class_count[y] += 1

			count_matrix[:, y] += x
			

		## Q2 - 1
		#print(self.classes)
		#print(class_count)
		#print("C(Si | time) = {0}".format(count_matrix[994, :]))
		#print("C(s_i | loss) = {0}".format(count_matrix[7596, :]))
		#print("C(s_i | export) = {0}".format(count_matrix[159, :]))
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

		score = [0] * len(self.classes)
		for i in xrange(0, len(self.classes)):
			W = self.weights[:, i]
			score[i] = np.dot(X, W)

		##Q2 - 3
		#print(self.classes)
		#print(score)

		pi = np.argmax(score)
		return self.classes[pi]
##NaiveBayes

class Perceptron:
	def __init__(self, feature, classes):

		self.feature_function = feature #BOWFeature(train_x, train_y)
		self.classes = classes #list(set(train_y))

		## Hack
		if self.feature_function.__class__.__name__ == 'BOWFeature':
			self.feature_function.binary = True
		elif self.feature_function.__class__.__name__ == 'LeskFeature':
			self.feature_function.binary = True


	def train(self, train_x, train_y):
		#print("Start Training:")

		k = self.feature_function.size()
		# Initialize weights to zeros
		self.weights = np.array([ ([0]*(k-1))+[1] for _ in xrange(0, len(self.classes))])
		self.weights = np.asfarray(self.weights)
		weights_sum = self.weights.copy()

		itt = 0
		for j in xrange(0, 3): 		## Itterations
			#print("Itteration {0}".format(j+1))
			for i in xrange(0, len(train_y)):
				#if(i % 300 == 0):
				#	print("{0:.2f}%".format(100 * (i/float(len(train_y)))))

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

		#print("Training is Done")


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
		#if(i % 100 == 0):
		#	print(i)

		c = NB.classify(test_texts[i])
		pred.append(c)
		gold.append(test_labels[i])

	#write_predictions(pred, 'q2p4.txt')

	return eval(gold, pred)

"""
Trains a perceptron model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""

def run_bow_perceptron_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	
	classes = list(set(train_labels))

	feature = BOWFeature(train_texts, train_labels)
	percept = Perceptron(feature, classes)
	percept.train(train_texts, train_labels)

	gold = []
	pred = []
	#print("Classify test set:")
	for i in xrange(0, len(test_texts)):
		#if(i % 300 == 0):
		#	print("{0:.2f}%".format(100 * (i/float(len(test_texts)))))

		c = percept.predict(test_texts[i])
		pred.append(c)
		gold.append(test_labels[i])
	#print("Classification Done")

	#write_predictions(pred, 'q3p3.txt')

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
	#feature = PCAFeature(train_texts, train_labels, 500)

	NB = NaiveBayes(feature, classes)
	NB.train(train_texts, train_labels)

	gold = []
	pred = []
	for i in xrange(0, len(test_texts)):
		#if(i % 100 == 0):
		#	print(i)

		c = NB.classify(test_texts[i])
		pred.append(c)
		gold.append(test_labels[i])

	#write_predictions(pred, 'q4p4_nb.txt')

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
	#feature = LeskFeature(classes)

	percept = Perceptron(feature, classes)
	percept.train(train_texts, train_labels)

	gold = []
	pred = []
	#print("Classify test set:")
	for i in xrange(0, len(test_texts)):
		#if(i % 300 == 0):
		#	print("{0:.2f}%".format(100 * (i/float(len(test_texts)))))

		c = percept.predict(test_texts[i])
		pred.append(c)
		gold.append(test_labels[i])
	#print("Classification Done")

	#write_predictions(pred, 'q4p4_pn.txt')

	return eval(gold, pred)


if __name__ == "__main__":
	# reading, tokenizing, and normalizing data
	train_labels, train_targets, train_texts = read_dataset('train')
	dev_labels, dev_targets, dev_texts = read_dataset('dev')
	test_labels, test_targets, test_texts = read_dataset('test')

	## running the classifier

	#test_scores = run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
	#           dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

	#test_scores = run_bow_perceptron_classifier(train_texts, train_targets, train_labels, 
	#		dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

	#test_scores = run_extended_bow_perceptron_classifier(train_texts, train_targets,train_labels, 
	#			dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)

	test_scores = run_extended_bow_naivebayes_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)

	print test_scores
