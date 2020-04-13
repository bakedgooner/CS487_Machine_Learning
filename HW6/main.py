# Cyrus Baker
# April 12, 2020
# HW6 - main.py

# dependencies
import time
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import regressors

dataset = sys.argv[1]
regressor = sys.argv[2]

if dataset not in ['housing', 'CRP']:
    sys.exit("Dataset not found")

if regressor not in ['lin_reg', 'ransac', 'ridge', 'lasso', 'norm_eq', 'svr']:
    sys.exit("Regressor not found")

if dataset == 'housing':
    print("\nHOUSING DATASET:\n")
    df = pd.read_csv("housing.data.txt", delim_whitespace=True, header=None)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_sc = StandardScaler()
    X_std = X_sc.fit_transform(X)

    y_sc = StandardScaler()
    y_std = y_sc.fit_transform(y[:, np.newaxis]).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    X_std_train, X_std_test, y_std_train, y_std_test = train_test_split(
        X_std, y_std, test_size=0.3, random_state=0)

else:
    print("\nCRP DATASET:\n")
    if regressor == 'norm_eq':
        sys.exit("Normal Equation only implemented on housing dataset")
    df = pd.read_csv("all_breakdown.csv", header=0)
    df.drop(columns="TIMESTAMP", inplace=True)
    df.dropna(axis=1, inplace=True)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_sc = StandardScaler()
    X_std = X_sc.fit_transform(X)

    y_sc = StandardScaler()
    y_std = y_sc.fit_transform(y[:, np.newaxis]).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    X_std_train, X_std_test, y_std_train, y_std_test = train_test_split(
        X_std, y_std, test_size=0.3, random_state=0)

if regressor == 'lin_reg':
    reg = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                random_state=1, alpha=1.0, gamma='auto', solver='auto', X_train=X_train, y_train=y_train, X_test=X_test)

    reg_std = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                    random_state=1, alpha=1.0, gamma='auto', solver='auto', X_train=X_std_train, y_train=y_std_train, X_test=X_std_test)

    print('non_standardized ', regressor)
    y_pred_test, y_pred_train = reg.lin_reg()
    print('\n\nstandardized ', regressor)
    y_std_pred_test, y_std_pred_train = reg_std.lin_reg()

    train_error = mean_squared_error(y_train, y_pred_train)
    test_error = mean_squared_error(y_test, y_pred_test)
    train_std_error = mean_squared_error(y_std_train, y_std_pred_train)
    test_std_error = mean_squared_error(y_std_test, y_std_pred_test)

    print("\n\ntrain_error: ", train_error)
    print("test_error: ", test_error)
    print("train_std_error: ", train_std_error)
    print("test_std_error: ", test_std_error)
    print("\n")

if regressor == 'ransac':
    reg = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                random_state=1, alpha=1.0, gamma='auto', solver='auto', X_train=X_train, y_train=y_train, X_test=X_test)

    reg_std = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                    random_state=1, alpha=1.0, gamma='auto', solver='auto', X_train=X_std_train, y_train=y_std_train, X_test=X_std_test)

    print('non_standardized ', regressor)
    y_pred_test, y_pred_train = reg.ransac()
    print('\n\nstandardized ', regressor)
    y_std_pred_test, y_std_pred_train = reg_std.ransac()

    train_error = mean_squared_error(y_train, y_pred_train)
    test_error = mean_squared_error(y_test, y_pred_test)
    train_std_error = mean_squared_error(y_std_train, y_std_pred_train)
    test_std_error = mean_squared_error(y_std_test, y_std_pred_test)

    print("\n\ntrain_error: ", train_error)
    print("test_error: ", test_error)
    print("train_std_error: ", train_std_error)
    print("test_std_error: ", test_std_error)
    print("\n")

if regressor == 'ridge':
    trer = []
    tser = []
    trstder = []
    tsstder = []

    for a in [0.9, 1.0, 1.5]:
        for s in ['auto', 'svd', 'lsqr']:
            reg = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                        random_state=1, alpha=a, gamma='auto', solver=s, X_train=X_train, y_train=y_train, X_test=X_test)

            reg_std = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                            random_state=1, alpha=a, gamma='auto', solver=s, X_train=X_std_train, y_train=y_std_train, X_test=X_std_test)

            print('non_standardized ', regressor)
            y_pred_test, y_pred_train = reg.ridge()
            print('\n\nstandardized ', regressor)
            y_std_pred_test, y_std_pred_train = reg_std.ridge()

            trer.append(mean_squared_error(y_train, y_pred_train))
            tser.append(mean_squared_error(y_test, y_pred_test))
            trstder.append(mean_squared_error(y_std_train, y_std_pred_train))
            tsstder.append(mean_squared_error(y_std_test, y_std_pred_test))

    train_error = min(trer)
    test_error = min(tser)
    train_std_error = min(trstder)
    test_std_error = min(tsstder)
    print("\n\ntrain_error: ", train_error)
    print("test_error: ", test_error)
    print("train_std_error: ", train_std_error)
    print("test_std_error: ", test_std_error)
    print("\n")


if regressor == 'lasso':
    trer = []
    tser = []
    trstder = []
    tsstder = []

    for a in [0.9, 1.0, 1.5]:
        reg = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                    random_state=1, alpha=a, gamma='auto', solver='auto', X_train=X_train, y_train=y_train, X_test=X_test)
        reg_std = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                        random_state=1, alpha=a, gamma='auto', solver='auto', X_train=X_std_train, y_train=y_std_train, X_test=X_std_test)
        print('non_standardized ', regressor)
        y_pred_test, y_pred_train = reg.lasso()
        print('\n\nstandardized ', regressor)
        y_std_pred_test, y_std_pred_train = reg_std.lasso()
        trer.append(mean_squared_error(y_train, y_pred_train))
        tser.append(mean_squared_error(y_test, y_pred_test))
        trstder.append(mean_squared_error(y_std_train, y_std_pred_train))
        tsstder.append(mean_squared_error(y_std_test, y_std_pred_test))

    train_error = min(trer)
    test_error = min(tser)
    train_std_error = min(trstder)
    test_std_error = min(tsstder)
    print("\n\ntrain_error: ", train_error)
    print("test_error: ", test_error)
    print("train_std_error: ", train_std_error)
    print("test_std_error: ", test_std_error)
    print("\n")

if regressor == 'norm_eq':
    reg = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                random_state=1, alpha=1.0, gamma='auto', solver='auto', X_train=X_train, y_train=y_train, X_test=X_test)

    reg_std = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                    random_state=1, alpha=1.0, gamma='auto', solver='auto', X_train=X_std_train, y_train=y_std_train, X_test=X_std_test)

    print('non_standardized ', regressor)
    y_pred_test, y_pred_train = reg.norm_eq()
    print('\n\nstandardized ', regressor)
    y_std_pred_test, y_std_pred_train = reg_std.norm_eq()

    train_error = mean_squared_error(y_train, y_pred_train)
    test_error = mean_squared_error(y_test, y_pred_test)
    train_std_error = mean_squared_error(y_std_train, y_std_pred_train)
    test_std_error = mean_squared_error(y_std_test, y_std_pred_test)

    print("\n\ntrain_error: ", train_error)
    print("test_error: ", test_error)
    print("train_std_error: ", train_std_error)
    print("test_std_error: ", test_std_error)
    print("\n")

if regressor == 'svr':
    trer = []
    tser = []
    trstder = []
    tsstder = []

    for g in ['auto', 'scale']:
        reg = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                    random_state=1, alpha=1.0, gamma='auto', solver='auto', X_train=X_train, y_train=y_train, X_test=X_test)
        reg_std = regressors.Regressors(min_samples=50, residual_threshold=5.0, max_trials=100,
                                        random_state=1, alpha=1.0, gamma='auto', solver='auto', X_train=X_std_train, y_train=y_std_train, X_test=X_std_test)
        print('non_standardized ', regressor)
        y_pred_test, y_pred_train = reg.svr()
        print('\n\nstandardized ', regressor)
        y_std_pred_test, y_std_pred_train = reg_std.svr()

        trer.append(mean_squared_error(y_train, y_pred_train))
        tser.append(mean_squared_error(y_test, y_pred_test))
        trstder.append(mean_squared_error(y_std_train, y_std_pred_train))
        tsstder.append(mean_squared_error(y_std_test, y_std_pred_test))

    train_error = min(trer)
    test_error = min(tser)
    train_std_error = min(trstder)
    test_std_error = min(tsstder)
    print("\n\ntrain_error: ", train_error)
    print("test_error: ", test_error)
    print("train_std_error: ", train_std_error)
    print("test_std_error: ", test_std_error)
    print("\n")
