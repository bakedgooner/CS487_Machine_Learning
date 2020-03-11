# Cyrus Baker
# March 9, 2020
# HW4 - classifiers.py

import sys
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier


class Classifiers(object):
    # Constructor

    def __init__(self, 
                 criterion="gini", 
                 max_depth=None, 
                 n_estimators=25, 
                 max_samples=1.0, 
                 min_samples_split=2.0, 
                 min_samples_leaf=1.0, 
                 max_features=1.0, 
                 bootstrap=True, 
                 learning_rate=0.1,
                 X_train=[], 
                 y_train=[], 
                 X_test=[]):
                 
        self.criterion = criterion
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.learning_rate = learning_rate
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.__obj = None

    def __fit(self):
        start = time.time() * 1000
        self.__obj.fit(self.X_train, self.y_train)
        end = time.time() * 1000
        score = self.__obj.score(self.X_train, self.y_train)
        runtime = end - start
        print("\nAccuracy for train: ", score)
        print("Running time for fit: " , runtime, " ms")
        return score, runtime

    def __predict(self):
        start = time.time() * 1000
        y_pred = self.__obj.predict(self.X_test)
        end = time.time() * 1000
        print("Running time for predict: " , end - start, " ms")
        return y_pred

    def decisiontree(self):
        self.__obj = DecisionTreeClassifier(criterion=self.criterion, 
                                            max_depth=self.max_depth)
        score = self.__fit()[0]
        runtime = self.__fit()[1]
        return self.__predict(), score, runtime
    
    def randforest(self):
        self.__obj = RandomForestClassifier(n_estimators=self.n_estimators, 
                                            criterion=self.criterion, 
                                            max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split, 
                                            min_samples_leaf=self.min_samples_leaf)
        score = self.__fit()[0]
        runtime = self.__fit()[1]
        return self.__predict(), score, runtime

    def bagging(self):
        tree = DecisionTreeClassifier()
        self.__obj = BaggingClassifier(base_estimator=tree, 
                                       n_estimators=self.n_estimators,
                                       max_samples=self.max_samples, 
                                       max_features=self.max_features,
                                       bootstrap=self.bootstrap)
        score = self.__fit()[0]
        runtime = self.__fit()[1]
        return self.__predict(), score, runtime

    def adaboost(self):
        tree = DecisionTreeClassifier()
        self.__obj = AdaBoostClassifier(base_estimator=tree,
                                        n_estimators=self.n_estimators, 
                                        learning_rate=self.learning_rate)
        score = self.__fit()[0]
        runtime = self.__fit()[1]
        return self.__predict(), score, runtime