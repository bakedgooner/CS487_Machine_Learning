# Cyrus Baker
# March 9, 2020
# HW 4 - main.py

import preprocess
import classifiers

import pandas as pd
import numpy as np
import sys
import time

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

classifier = sys.argv[1]
dataset = sys.argv[2]

# check that classifier is valid
if classifier not in ["decisiontree", "randforest", "bagging", "adaboost"]:
    sys.exit("Classifier not found")

# check that dataset is valid
if dataset not in ["digits", "mammographic_masses.data"]:
    sys.exit("Dataset not found")

if dataset == "digits":
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

else: 
    fileName = str(dataset)
    dataset = preprocess.preprocess(fileName)
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

if classifier == "decisiontree":
    maxAcc = 0
    for c in ["gini", "entropy"]:
        for m in [5, 10, 20]:
            print("\ncriterion = ", c)
            print("max_depth = ", m)
            classify = classifiers.Classifiers(criterion=c, 
                                               max_depth=m, 
                                               X_train=X_train_std, 
                                               y_train=y_train, 
                                               X_test=X_test_std)
            start = time.time() * 1000
            y_pred = classify.decisiontree()[0]
            fitScore = classify.decisiontree()[1]
            fitTime = classify.decisiontree()[2]
            end = time.time() * 1000
            acc = accuracy_score(y_test, y_pred)
            runtime = end - start
            if acc > maxAcc:
                maxAcc = acc
                bestTime = runtime
                bestCriterion = c
                bestMaxDepth = m
                bestFitScore = fitScore
                bestFitTime = fitTime
            print("Running time: " , runtime, " ms")
            print('Accuracy: %.4f' % acc)

    print("\nHighest accuracy = ", maxAcc)
    print("time = ", bestTime)
    print("criterion = ", bestCriterion)
    print("max depth = ", bestMaxDepth)
    print("bestFitScore = ", bestFitScore)
    print("bestFitTime = ", bestFitTime)



elif classifier == "randforest":

    """For Random Forest, compare what happens with at least three different values for: 
    n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf."""

    maxAcc = 0

    for n in [10, 50, 100]:
        for c in ["gini", "entropy"]:
            for m in [5, 32, None]:
                for mn in [2, 5, 10]:
                    for ml in [1, 5, 10]:
                        print("n_estimators = ", n)
                        print("criterion = ", c)
                        print("max_depth = ", m)
                        print("min_samples_split = ", mn)
                        print("min_samples_leaf= ", ml)
                        classify = classifiers.Classifiers(n_estimators=n, 
                                                           criterion=c, 
                                                           max_depth=m, 
                                                           min_samples_split=mn, 
                                                           min_samples_leaf=ml, 
                                                           X_train=X_train_std, 
                                                           y_train=y_train, 
                                                           X_test=X_test_std)
                        start = time.time() * 1000
                        y_pred = classify.randforest()[0]
                        fitScore = classify.randforest()[1]
                        fitTime = classify.randforest()[2]
                        end = time.time() * 1000
                        acc = accuracy_score(y_test, y_pred)
                        runtime = end - start
                        if acc > maxAcc:
                            maxAcc = acc
                            bestTime = runtime
                            bestNEst = n
                            bestCriterion = c
                            bestMaxDepth = m
                            bestMinSplit = mn
                            bestMinLeaf = ml
                            bestFitScore = fitScore
                            bestFitTime = fitTime
                        print("Running time: " , runtime, " ms")
                        print('Accuracy: %.4f' % acc)

    print("\nmaxAcc = ", maxAcc)
    print("bestTime = ", bestTime)
    print("bestNEst = ", bestNEst)
    print("bestCriterion = ", bestCriterion)
    print("bestMaxDepth = ", bestMaxDepth)
    print("bestMinSplit = ", bestMinSplit)
    print("bestMinLeaf = ", bestMinLeaf)
    print("bestFitScore = ", bestFitScore)
    print("bestFitTime = ", bestFitTime)

elif classifier == "bagging":

    """For Bagging, at least three different values for: 
    n_estimators, max_samples, max_features, bootstrap (True | False)."""

    maxAcc = 0

    for n in [10, 50, 100]:
        for m in [1, 5, 10]:
            for mx in [1, 2, 3]:
                for b in [True, False]:
                    print("\nn_estimators = ", n)
                    print("max_samples = ", m)
                    print("max_features = ", mx)
                    print("bootstrap = ", b)
                    classify = classifiers.Classifiers(n_estimators=n, 
                                                       max_samples=m, 
                                                       max_features=mx, 
                                                       bootstrap=b, 
                                                       X_train=X_train_std, 
                                                       y_train=y_train, 
                                                       X_test=X_test_std)
                    start = time.time() * 1000
                    y_pred = classify.bagging()[0]
                    fitScore = classify.bagging()[1]
                    fitTime = classify.bagging()[2]
                    end = time.time() * 1000
                    acc = accuracy_score(y_test, y_pred)
                    runtime = end - start
                    if acc > maxAcc:
                        maxAcc = acc
                        bestNEst = n
                        bestMaxSamples = m
                        bestMaxFeatures = mx
                        bestBoot = b
                        bestFitScore = fitScore
                        bestFitTime = fitTime
                    print("Running time: " , runtime, " ms")
                    print('Accuracy: %.4f' % acc)

    print("\nmaxAcc = ", maxAcc)
    print("bestNEst = ", bestNEst)
    print("bestMaxSamples = ", bestMaxSamples)
    print("bestMaxFeatures = ", bestMaxFeatures)
    print("bestBoot = ", bestBoot)
    print("bestFitScore = ", bestFitScore)
    print("bestFitTime = ", bestFitTime)

elif classifier == "adaboost":

    """For AdaBoost, at least three values for: 
    n_estimators, learning_rate. You can use tables to
    show the results of each comparison"""

    maxAcc = 0

    for n in [10, 50, 100]:
        for l in [0.01, 0.1, 0.2]:
            print("\nn_estimators = ", n, " learning_rate = ", l)
            classify = classifiers.Classifiers(n_estimators=n, 
                                               learning_rate=l, 
                                               X_train=X_train_std, 
                                               y_train=y_train, 
                                               X_test=X_test_std)
            start = time.time() * 1000
            y_pred = classify.adaboost()[0]
            fitScore = classify.adaboost()[1]
            fitTime = classify.adaboost()[2]
            end = time.time() * 1000
            acc = accuracy_score(y_test, y_pred)
            runtime = end - start
            if acc > maxAcc:
                maxAcc = acc
                bestTime = runtime
                bestNEst = n
                bestLearnRate = l
                bestFitScore = fitScore
                bestFitTime = fitTime
            print("Running time: " , runtime, " ms")
            print('Accuracy: %.4f' % acc)
    print("\nmaxAcc = ", maxAcc)
    print("bestTime = ", bestTime)
    print("bestNEst = ", bestNEst)
    print("bestLearnRate = ", bestLearnRate)
    print("bestFitScore = ", bestFitScore)
    print("bestFitTime = ", bestFitTime)