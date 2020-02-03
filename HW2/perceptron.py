# Cyrus Baker
# February 3, 2020
# perceptron.py

# dependencies
import numpy as np

# Design and implement a Perceptron binary classifier.
class Perceptron(object):
    
    def __init__(self, learning_rate=0.01, num_iter=50, rand_state=1):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.rand_state = rand_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.rand_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
        self.errors_ = []
            
        for _ in range(self.num_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
        return self
                
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
