import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def readTxt(path):
	return np.genfromtxt(
		path, 
    delimiter=',', 
    dtype=[('name','U20'), ('gender','S1'),('count','i4')],
    converters={0: lambda s:s.decode('utf-8').lower()})	


my_data = np.genfromtxt(
    '../textdata/name_gender.csv',
    delimiter=',',
    dtype=[('name','U20'),('gender','S1')],
    converters={0:lambda x: x.decode('utf-8').lower()}
    )

my_data = np.array(list(filter(lambda x:not (x[1].decode('utf-8').startswith('?') or x[1].decode('utf-8').startswith('1')),my_data)),dtype=[('name','U20'),('gender','S1')],)

print(len(my_data))
male = 0
female = 0
for gender in my_data['gender']:
	if gender.decode('utf-8') == 'M':
		male+=1
	if gender.decode('utf-8') == 'F':
		female+=1


print(male)
print(female)
