# Cyrus Baker
# main.py
# 03/01/2020

# Dependencies
import classifiers
import preprocess

import sys
import pandas as pd
import numpy as np
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


classifier = sys.argv[1]
dataset = sys.argv[2]

# check that classifier is valid
if classifier not in ["perceptron", "svm_l", "svm_nl", "dt_gini", "dt_entropy", "knn_minkowski", "knn_euclidean", "knn_manhattan"]:
    sys.exit("Classifier not found")

# check that dataset is valid
if dataset not in ["digits", "eighthr"]:
    sys.exit("Dataset not found")

# get dataset
if dataset == "digits":
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
# must be eighthr
else: 
    eighthr = preprocess.preprocess()
    X = eighthr.iloc[:,:-1].values
    y = eighthr.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

if classifier == "perceptron":
    classify = classifiers.Classifiers(eta=0.001, random_state=1, X_train=X_train_std, y_train=y_train, X_test=X_test_std)
    start = time.time() * 1000
    y_pred = classify.perceptron()
    end = time.time() * 1000
    print("Running time: " , end - start, " ms")
    print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
elif classifier == "svm_l":
    classify = classifiers.Classifiers(X_train=X_train, y_train=y_train, X_test=X_test)
    start = time.time() * 1000
    y_pred = classify.svm_l()
    end = time.time() * 1000
    print("Running time: " , end - start, " ms")
    print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
elif classifier == "svm_nl":
    classify = classifiers.Classifiers(random_state=1, C=1, X_train=X_train, y_train=y_train, X_test=X_test)
    start = time.time() * 1000
    y_pred = classify.svm_nl()
    end = time.time() * 1000
    print("Running time: " , end - start, " ms")
    print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
elif classifier == "dt_gini":
    for n in range(2, 11):
        classify = classifiers.Classifiers(random_state=1, max_depth=n, X_train=X_train, y_train=y_train, X_test=X_test)
        start = time.time() * 1000
        y_pred = classify.dt_gini()
        end = time.time() * 1000
        print("Running time: " , end - start, " ms")
        print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
elif classifier == "dt_entropy":
    for n in range(2, 11):
        classify = classifiers.Classifiers(random_state=1, max_depth=n, X_train=X_train, y_train=y_train, X_test=X_test)
        start = time.time() * 1000
        y_pred = classify.dt_entropy()
        end = time.time() * 1000
        print("Running time ", n, ":" , end - start, " ms")
        print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
elif classifier == "knn_minkowski":
    for n in range(1, 11):
        classify = classifiers.Classifiers(p=1, n_neighbors=n, X_train=X_train, y_train=y_train, X_test=X_test)
        start = time.time() * 1000
        y_pred = classify.knn_minkowski()
        end = time.time() * 1000
        print("Running time ", n, ":" , end - start, " ms")
        print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
elif classifier == "knn_euclidean":
    for n in range(1, 11):
        classify = classifiers.Classifiers(p=1, n_neighbors=n, X_train=X_train, y_train=y_train, X_test=X_test)
        start = time.time() * 1000
        y_pred = classify.knn_euclidean()
        end = time.time() * 1000
        print("Running time ", n, ":" , end - start, " ms")
        print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
elif classifier == "knn_manhattan":
    for n in range(1, 11):
        classify = classifiers.Classifiers(p=1, n_neighbors=n, X_train=X_train, y_train=y_train, X_test=X_test)
        start = time.time() * 1000
        y_pred = classify.knn_manhattan()
        end = time.time() * 1000
        print("Running time ", n, ":" , end - start, " ms")
        print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))