"""
CMSC723 / INST725 / LING723 -- Fall 2016
Project 1: Implementing Word Sense Disambiguation Systems
"""


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
from corpus import Corpus
import numpy as np
def run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	
	corpus = Corpus(train_labels, train_texts)

	print('Examples: {0}'.format(len(train_labels)))

	print(corpus.classes)
	M = np.array(corpus.matrix)

	w1 = corpus.vocabulary.index('time')
	w2 = corpus.vocabulary.index('loss')
	w3 = corpus.vocabulary.index('export')

	print('c(s) {0}'.format(np.sum(M, axis=0)))
	print('c(s) {0}'.format(corpus.class_count))

	print('c(s, time) {0}'.format(M[w1, :]))
	print('c(s, loss) {0}'.format(M[w2, :]))
	print('c(s, export) {0}'.format(M[w3, :]))

	print('p(s) {0}'.format(np.divide(corpus.class_count, float(corpus.N))))
	print('p(time) {0}'.format(np.sum(M[w1, :]) / float(np.sum(M))))
	print('p(loss) {0}'.format(np.sum(M[w2, :]) / float(np.sum(M))))
	print('p(export) {0}'.format(np.sum(M[w3, :]) / float(np.sum(M))))

	print('p(time | s) {0}'.format(np.sum(M[w3, :]) / float(np.sum(M))))
	print('p(loss | s) {0}'.format(np.sum(M[w3, :]) / float(np.sum(M))))
	print('p(export | s) {0}'.format(np.sum(M[w3, :]) / float(np.sum(M))))

	np.set_printoptions(threshold='nan')

	W_0 = corpus.get_weights(corpus.classes[0]) #cord
	W_1 = corpus.get_weights(corpus.classes[1]) #division
	W_2 = corpus.get_weights(corpus.classes[2]) #product
	W_3 = corpus.get_weights(corpus.classes[3]) #text
	W_4 = corpus.get_weights(corpus.classes[4]) #phone
	W_5 = corpus.get_weights(corpus.classes[5]) #formation

	for i in xrange(0, 20):
		X = corpus.get_bow_vector(train_texts[i])
		X = np.append(X, 1)

		score_xi_w0 = np.dot(X, W_0)
		score_xi_w1 = np.dot(X, W_1)
		score_xi_w2 = np.dot(X, W_2) 
		score_xi_w3 = np.dot(X, W_3)
		score_xi_w4 = np.dot(X, W_4)
		score_xi_w5 = np.dot(X, W_5)
		#print('score_X{0}_cord: {1}'.format(i, score_xi_w0))
		#print('score_X{0}_division: {1}'.format(i, score_xi_w1))
		#print('score_X{0}_product: {1}'.format(i, score_xi_w2))
		#print('score_X{0}_text: {1}'.format(i, score_xi_w3))
		#print('score_X{0}_phone: {1}'.format(i, score_xi_w4))
		#print('score_X{0}_formation: {1}'.format(i, score_xi_w5))

		pi = np.argmax([score_xi_w0, score_xi_w1, score_xi_w2, score_xi_w3, score_xi_w4, score_xi_w5])
		print('Y{0}: {1}, Predict: {2}'.format(i, train_labels[i], corpus.classes[pi]))

	#print('X_0 {0}'.format(X))
	#print('W_cord {0}'.format(W) )


	"""
	**Your final classifier implementation of part 2 goes here**
	"""
	pass

"""
Trains a perceptron model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_bow_perceptron_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	"""
	**Your final classifier implementation of part 3 goes here**
	"""
	pass



"""
Trains a naive bayes model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_naivebayes_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	"""
	**Your final implementation of Part 4 with perceptron classifier**
	"""
	pass

"""
Trains a perceptron model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_perceptron_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	"""
	**Your final implementation of Part 4 with perceptron classifier**
	"""
	pass


def run_naivebayes_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	

	labels = train_labels + dev_labels
	num_labels = len(labels)
	count_dict = {}

	for w in labels:
		if w in count_dict:
			count_dict[w] = count_dict[w] +1
		else:
			count_dict[w] = 1

	priors = []
	print('Data Set Priors:')
	for k,v in count_dict.items():
		print('{0} : ({1}/{2}) {3}'.format(k, v, num_labels, float(v)/num_labels))
		priors.append(float(v)/num_labels)

	return max(priors)


if __name__ == "__main__":
    # reading, tokenizing, and normalizing data
    train_labels, train_targets, train_texts = read_dataset('train')
    dev_labels, dev_targets, dev_texts = read_dataset('dev')
    test_labels, test_targets, test_texts = read_dataset('test')

    #running the classifier
    #test_scores = run_naivebayes_classifier(train_texts, train_targets, train_labels, 
	#			dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

    test_scores = run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
				dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

    print test_scores
