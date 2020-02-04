# Cyrus Baker
# February 3, 2020
# adaline.py

# dependencies
import numpy as np

# (20 points) Design and implement an Adaline binary classifier.

class Adaline(object):
    def __init__(self, learning_rate=0.01, num_iter=50, rand_state=1):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.rand_state = rand_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.rand_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
        size=1 + X.shape[1])
        self.cost_ = []
        for i in range(self.num_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.learning_rate * X.T.dot(errors)
            self.w_[0] += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
