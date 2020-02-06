# Cyrus Baker
# February 3, 2020
# adaline.py

# dependencies
import numpy as np

class Adaline(object):
    def __init__(self, rate = 0.01, niter = 10):
        self.rate = rate
        self.niter = niter   
    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        """  
        # weights
        self.weight = np.zeros(1 + X.shape[1])   
        # Number of misclassifications
        self.errors = [] 
        # Cost function
        self.cost = []   
        for i in range(self.niter):
            output = self.net_input(X)
            errors = y - output
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
        return self  
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weight[1:]) + self.weight[0]   
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X) 
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)    