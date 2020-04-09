# Cyrus Baker
# March 30, 2020
# HW5 - main.py

# Dependencies
import pandas as pd
import numpy as np
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import algorithms

dataset = sys.argv[1]
algorithm = sys.argv[2]

if dataset not in ["iris", "MNIST"]:
    sys.exit("Dataset not found")

if algorithm not in ["PCA", "LDA", "Kernel_PCA"]:
    sys.exit("Algorithm not found")

if dataset == "iris":
    df = pd.read_csv("iris.csv", header=None)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

else:
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=5000, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


if algorithm == 'PCA':
    print("PCA:\n")
    max_acc = 0
    max_prec = 0
    max_rec = 0
    max_f1 = 0

    for n in [2, 3]:
        alg = algorithms.Algorithms(n_components=n, kernel="rbf", gamma=15, max_depth=5, criterion='gini',
                                    C=100, random_state=1, X_train=X_train, y_train=y_train, X_test=X_test)
        X_test_red, X_train_red = alg.run_pca()
        y_pred_pre = alg.decisiontree()
        print('Acc_pre: ', accuracy_score(y_test, y_pred_pre))
        print('Pre_pre: ', precision_score(
            y_test, y_pred_pre, average='micro'))
        print('rec_pre: ', recall_score(y_test, y_pred_pre, average='micro'))
        print('f1_pre: ', f1_score(y_test, y_pred_pre, average='micro'))
        start = time.time() * 1000
        alg.X_train = X_train_red
        alg.X_test = X_test_red
        y_pred = alg.decisiontree()
        end = time.time() * 1000
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='micro')
        runtime = end - start
        if acc > max_acc:
            max_acc = acc
        if precision > max_prec:
            max_prec = precision
        if recall > max_rec:
            max_rec = recall
        if f1 > max_f1:
            max_f1 = f1
        print("Running time: ", runtime, " ms")

    print('MaxAccuracy: %.4f' % max_acc)
    print('MaxPrecision: %.4f' % max_prec)
    print('MaxRecall: %.4f' % max_rec)
    print('MaxF1: %.4f' % max_f1)

if algorithm == 'LDA':
    print("LDA:\n")
    max_acc = 0
    max_prec = 0
    max_rec = 0
    max_f1 = 0
    for n in [2, 3]:
        alg = algorithms.Algorithms(n_components=n, kernel="rbf", gamma=15, max_depth=5, criterion='gini',
                                    C=100, random_state=1, X_train=X_train, y_train=y_train, X_test=X_test)
        X_test_red, X_train_red = alg.run_lda()
        y_pred_pre = alg.decisiontree()
        print('Acc_pre: ', accuracy_score(y_test, y_pred_pre))
        print('Pre_pre: ', precision_score(
            y_test, y_pred_pre, average='micro'))
        print('rec_pre: ', recall_score(y_test, y_pred_pre, average='micro'))
        print('f1_pre: ', f1_score(y_test, y_pred_pre, average='micro'))
        start = time.time() * 1000
        alg.X_train = X_train_red
        alg.X_test = X_test_red
        y_pred = alg.decisiontree()
        end = time.time() * 1000
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='micro')
        runtime = end - start
        if acc > max_acc:
            max_acc = acc
        if precision > max_prec:
            max_prec = precision
        if recall > max_rec:
            max_rec = recall
        if f1 > max_f1:
            max_f1 = f1
        print("Running time: ", runtime, " ms")
        print('Accuracy: %.4f' % acc)
        print('Precision: %.4f' % precision)
        print('Recall: %.4f' % recall)
        print('F1: %.4f' % f1)

    print('MaxAccuracy: %.4f' % max_acc)
    print('MaxPrecision: %.4f' % max_prec)
    print('MaxRecall: %.4f' % max_rec)
    print('MaxF1: %.4f' % max_f1)

if algorithm == 'Kernel_PCA':
    print("Kernel_PCA:\n")
    max_acc = 0
    max_prec = 0
    max_rec = 0
    max_f1 = 0
    for k in ["rbf", "poly", "sigmoid"]:
        print(k)
        for n in [2, 3]:

            alg = algorithms.Algorithms(n_components=n, kernel=k, gamma=15, max_depth=5, criterion='gini',
                                        C=100, random_state=1, X_train=X_train, y_train=y_train, X_test=X_test)
            X_test_red, X_train_red = alg.run_kernel_pca()
            y_pred_pre = alg.decisiontree()
            print('Acc_pre: ', accuracy_score(y_test, y_pred_pre))
            print('Pre_pre: ', precision_score(
                y_test, y_pred_pre, average='micro'))
            print('rec_pre: ', recall_score(
                y_test, y_pred_pre, average='micro'))
            print('f1_pre: ', f1_score(y_test, y_pred_pre, average='micro'))
            start = time.time() * 1000
            alg.X_train = X_train_red
            alg.X_test = X_test_red
            y_pred = alg.decisiontree()
            end = time.time() * 1000
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='micro')
            recall = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='micro')
            runtime = end - start
            if acc > max_acc:
                max_acc = acc
            if precision > max_prec:
                max_prec = precision
            if recall > max_rec:
                max_rec = recall
            if f1 > max_f1:
                max_f1 = f1
            print("Running time: ", runtime, " ms")
            print('Accuracy: %.4f' % acc)
            print('Precision: %.4f' % precision)
            print('Recall: %.4f' % recall)
            print('F1: %.4f' % f1)

    print('MaxAccuracy: %.4f' % max_acc)
    print('MaxPrecision: %.4f' % max_prec)
    print('MaxRecall: %.4f' % max_rec)
    print('MaxF1: %.4f' % max_f1)
