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

def toDict(name):
    d = {}
    d['last_three'] = name[-3:]
    d['first_two'] = name[:2]
    d['last_two'] = name[-2:]
    d['last_one'] = name[-1:]
    return d

vec1 = DictVectorizer()

name_map = np.vectorize(toDict, otypes=[np.ndarray])

X = name_map(my_data['name']).tolist()

X = vec1.fit_transform(X).toarray()

y = my_data['gender']

test_data = np.genfromtxt(
    '../textdata/testset2.txt',
    delimiter=',',
    dtype=[('name','U20'),('gender','S1'),('_id','S30')],
    converters={0:lambda x: x.decode('utf-8').lower()}
    )

Xt = name_map(test_data['name']).tolist()

Xt = vec1.transform(Xt).toarray()

yt = test_data['gender']

rf_model = RandomForestClassifier(random_state=123456,n_estimators=30)

params_grid={
'n_estimators': [10,30,50]
}

# svm_model = GridSearchCV(SVC(C=1),params_grid,n_jobs=4,cv=5,verbose=3,scoring='precision_macro')

# print(svm_model.get_params())
CV_rfc = GridSearchCV(estimator=rf_model, param_grid=params_grid,verbose=3,cv=3)
CV_rfc.fit(X, y)

print(CV_rfc.best_estimator_)

print(classification_report(yt,CV_rfc.predict(Xt)))
