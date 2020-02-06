# Cyrus Baker
# February 3, 2020
# sgd.py

# Dependencies
import numpy as np
from numpy.random import seed

# (20 points) Design and implement a Stochastic Gradient Descent (SGD) binary classifier.

class SGD(object):
    def __init__(self, rate = 0.01, niter = 10, shuffle=True, random_state=None):
        self.rate = rate
        self.niter = niter
        self.weight_initialized = False  
        # If True, Shuffles training data every epoch
        self.shuffle = shuffle   
        # Set random state for shuffling and initializing the weights.
        if random_state:
            seed(random_state)    
    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        """  
        # weights
        self.initialize_weights(X.shape[1])  
        # Cost function
        self.cost = []   
        for i in range(self.niter):
            if self.shuffle:
                X, y = self.shuffle_set(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self.update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost.append(avg_cost)
        return self  
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.weight_initialized:
            self.initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.update_weights(xi, target)
        else:
            self.up
        return self  
    def shuffle_set(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]    
    def initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.weight = np.zeros(1 + m)
        self.weight_initialized = True   
    def update_weights(self, xi, target):
        """Apply SGD learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.weight[1:] += self.rate * xi.dot(error)
        self.weight[0] += self.rate * error
        cost = 0.5 * error**2
        return cost  
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weight[1:]) + self.weight[0]   
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X) 
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)    