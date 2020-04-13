# Cyrus Baker
# April 12, 2020
# HW6 - regressors.py

# dependencies
import time

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

import numpy as np


class Regressors(object):
    def __init__(self,
                 min_samples,
                 residual_threshold,
                 max_trials,
                 random_state,
                 alpha,
                 gamma,
                 solver,
                 X_train=[],
                 y_train=[],
                 X_test=[]):
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.random_state = random_state
        self.alpha = alpha
        self.gamma = gamma
        self.solver = solver
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
        print("Running time for fit: ", runtime, " ms")

    def __predict(self):
        start = time.time() * 1000
        y_pred_test = self.__obj.predict(self.X_test)
        end = time.time() * 1000
        print("Running time for y_pred_test: ", end - start, " ms")

        start = time.time() * 1000
        y_pred_train = self.__obj.predict(self.X_train)
        end = time.time() * 1000
        print("Running time for y_pred_train: ", end - start, " ms")

        return y_pred_test, y_pred_train

    def lin_reg(self):
        self.__obj = LinearRegression()
        self.__fit()
        return self.__predict()

    def ransac(self):
        self.__obj = RANSACRegressor(LinearRegression(), min_samples=self.min_samples,
                                     residual_threshold=self.residual_threshold, max_trials=self.max_trials, random_state=self.random_state)
        self.__fit()
        return self.__predict()

    def ridge(self):
        self.__obj = Ridge(alpha=self.alpha, solver=self.solver,
                           random_state=self.random_state)
        self.__fit()
        return self.__predict()

    def lasso(self):
        self.__obj = Lasso(alpha=self.alpha, random_state=self.random_state)
        self.__fit()
        return self.__predict()

    def norm_eq(self):
        start = time.time() * 1000
        bias_vec = np.ones((self.X_train.shape[0]))
        bias_vec = bias_vec[:, np.newaxis]
        X_bias = np.hstack((bias_vec, self.X_train))

        theta = np.zeros(self.X_train.shape[1])
        temp = np.linalg.inv(np.dot(X_bias.T, X_bias))
        theta = np.dot(temp, np.dot(X_bias.T, self.y_train))

        end = time.time() * 1000
        print("Running time for norm_eq training: ", end - start, " ms")

        start = time.time() * 1000
        y_pred_train = np.dot(self.X_train, theta[1:]) + theta[0]
        end = time.time() * 1000
        print("Running time for norm_eq y_pred_train: ", end - start, " ms")

        start = time.time() * 1000
        y_pred_test = np.dot(self.X_test, theta[1:]) + theta[0]
        end = time.time() * 1000
        print("Running time for norm_eq y_pred_test: ", end - start, " ms")

        return y_pred_test, y_pred_train

    def svr(self):
        self.__obj = SVR(gamma=self.gamma)
        self.__fit()
        return self.__predict()
