import nltk_nb as nb
import numpy as np
import sys

trainset = []

with open(sys.argv[1],'r') as f:
	for line in f:
		name, gender,_id= line.strip('\n').split(',')
		trainset.append([name,gender])

nb_count_male = 0
nb_count_female = 0

female_count = 0
male_count = 0

for name,gender in trainset:
	if gender == 'M':
		male_count+=1
	else:
		female_count+=1

	if nb.guess(name) == gender:
		if gender == 'M':
			nb_count_male+=1
		else:
			nb_count_female+=1

print('NaiveBayes Female: %s'%(nb_count_female/female_count))
print('NaiveBayes Male`: %s'%(nb_count_male/male_count))

all_count = female_count+male_count

print()
print('NaiveBayes Overall: %s'%((nb_count_female+nb_count_male)/all_count))



