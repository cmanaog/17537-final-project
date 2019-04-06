# Decision Tree model for hmda data

import sys, time, os
import numpy as np 
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.svm import LinearSVC

from sklearn.preprocessing import normalize
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

from joblib import dump, load

print "-- fetching data ..."

# train_data = pd.read_csv("../Downloads/train_clean.csv")
# x_train = np.asarray(train_data.iloc[:, 1:-1], dtype = np.float64)
# y_train = np.asarray(train_data.iloc[:, -1], dtype = np.float64)

# np.save('x_train.npy', x_train)
# np.save('y_train.npy', y_train)

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

print "read " + str(len(y_train)) + " train samples"

# test_data = pd.read_csv("../Downloads/test_clean.csv")
# x_test = np.asarray(test_data.iloc[:, 1:-1], dtype = np.float64)
# y_test = np.asarray(test_data.iloc[:, -1], dtype = np.float64)

# np.save('x_test.npy', x_test)
# np.save('y_test.npy', y_test)

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

print "read " + str(len(y_test)) + " test samples"


print "-- machine learning ..."

# various classifiers to experiment with:

rfclf = RandomForestClassifier(n_estimators=10).fit(x_train, y_train)
dump(rfclf, 'rfclf.joblib') 
# rfclf = load('rfclf.joblib') 
print "RF score: " + str(rfclf.score(x_test, y_test))

gbclf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=24).fit(x_train, y_train)
dump(gbclf, 'gbclf.joblib') 
# gbclf = load('gbclf.joblib') 
print "GB score: " + str(gbclf.score(x_test, y_test))

knclf = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
dump(knclf, 'knclf.joblib') 
# knclf = load('knclf.joblib') 
print "KN score: " + str(knclf.score(x_test, y_test))


print "-- done."
