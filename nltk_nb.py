from nltk.corpus import names
import nltk
import random
from pymongo import MongoClient

def get_gender_features(word):
	return {
	'last_one':word[-1],
	# 'first_one':word[0],
	# 'first_two':word[:2],
	# 'first_three':word[:3],
	'last_two':word[-2:],
	# 'length':len(word),
	'last_three':word[-3:],
	}

# labeled_names = ([(name, 'M') for name in names.words('male.txt')]	+ [(name, 'F') for name in names.words('female.txt')])

def read(filename):
	arr = []
	with open(filename,'r',encoding='utf-8') as f:
		for line in f:
			name, gender,count= line.strip('\n').split(',')
			arr.append([name,gender])

	return arr

labeled_names = read('../textdata/names/yob2000.txt')+read('../textdata/names/yob2012.txt')
# +read('textdata/names/yob1995.txt')

# labeled_names = read('../textdata/name_gender.csv')
# labeled_names = list(filter(lambda x:not x[1].startswith('?'),labeled_names))

feature_sets = [(get_gender_features(name),gender) for (name,gender) in labeled_names]
train_set = feature_sets

nb_classifier = nltk.NaiveBayesClassifier.train(train_set)

def guess(word):
	return nb_classifier.classify(get_gender_features(word))

