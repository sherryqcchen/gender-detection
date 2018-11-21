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

rf_model = RandomForestClassifier(random_state=123456,n_estimators=50)

knn_model = KNeighborsClassifier(n_jobs=4,n_neighbors=10)

# print(classification_report(yt,rf_model.predict(Xt)))

model = rf_model

model.fit(X, y)

# actual_arr = [(lambda x:1 if x.decode('utf-8') =='M' else 0)(x) for x in yt]
# prediction_arr = [x[1] for x in model.predict_proba(Xt)]

# false_positive_rate, true_positive_rate, thresholds = roc_curve(actual_arr,prediction_arr)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# fig,ax = plt.subplots()
# plt.title('Receiver Operating Characteristic')
# plt.plot(false_positive_rate, true_positive_rate, 'b',
# label='AUC = %0.2f'% roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0,1],[0,1],'r--')
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# # plt.show()  
# fig.savefig('rf_roc.eps',format='eps',dpi=1200)


def fit_and_get_proba(data):
    Xt=vec1.transform(name_map(data['name']).tolist()).toarray()
    _ids = data['_id']
    probabilities = model.predict_proba(Xt)
    d = {}
    for i in range(len(_ids)):
        d[_ids[i].decode('utf-8')] = probabilities[i][1]
    print('one model ready.')
    return d


