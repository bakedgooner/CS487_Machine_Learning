# Cyrus Baker
# classifiers.py
# 03/01/2020

import sys
import time
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# create and run classifiers
class Classifiers(object):
    # constructor
    def __init__(self, eta=0.1, iters=10, random_state=1, C=1, max_depth=2, n_neighbors=1, p=1, X_train=[], y_train=[], X_test=[]):
        self.eta = eta
        self.iters = iters
        self.random_state = random_state
        self.C = C
        self.max_depth = max_depth
        self.n_neighbors = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.__obj = None

    def __fit(self):
        start = time.time() * 1000
        self.__obj.fit(self.X_train, self.y_train)
        end = time.time() * 1000
        print("\nAccuracy for train: ", self.__obj.score(self.X_train, self.y_train))
        print("Running time for fit: " , end - start, " ms")

    def __predict(self):
        start = time.time() * 1000
        y_pred = self.__obj.predict(self.X_test)
        end = time.time() * 1000
        print("Running time for predict: " , end - start, " ms")
        return y_pred

    def perceptron(self):
        self.__obj = Perceptron(eta0=self.eta, random_state=self.random_state)
        self.__fit()
        return self.__predict()

    def svm_l(self):
        self.__obj = SVC(kernel="linear")
        self.__fit()
        return self.__predict()

    def svm_nl(self):
        self.__obj = SVC(kernel="rbf", random_state=self.random_state, C=self.C)
        self.__fit()
        return self.__predict()

    def dt_gini(self):
        self.__obj = DecisionTreeClassifier(criterion="gini", random_state=self.random_state, max_depth=self.max_depth)
        self.__fit()
        return self.__predict()
    
    def dt_entropy(self):
        self.__obj = DecisionTreeClassifier(criterion="entropy", random_state=self.random_state, max_depth=self.max_depth)
        self.__fit()
        return self.__predict()

    def knn_minkowski(self):
        self.__obj = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="minkowski", p=self.p)
        self.__fit()
        return self.__predict()
    def knn_euclidean(self):
        self.__obj = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="euclidean", p=self.p)
        self.__fit()
        return self.__predict()
    def knn_manhattan(self):
        self.__obj = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="manhattan", p=self.p)
        self.__fit()
        return self.__predict()