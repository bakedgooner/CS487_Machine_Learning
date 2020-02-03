# Cyrus Baker
# February 3, 2020
# HW2.py

# Dependencies
import numpy as np
from adaline import Adaline
from perceptron import Perceptron
from sgd import SGD

# (20 points) Write program to test the different classifiers.


#   (a) (1 point) Your program should be called main.py.


#  (b) (2 points) It should have two arguments: (i) classifier name, which can be ‘perceptron’, ‘adaline’, and ‘sgd’, and (ii) data file (including path information). You can have other arguments depending on the design of your program.


#  (c) (2 points) It should have proper error checking functions (e.g., make sure the classifier name is a valid one).


# (d) (15 points) It calls the different classifiers to train models, make predictions, and report prediction errors.


# 5. (18 points) Write a report report.pdf to analyze the predictive power and the running time of different classifiers.

#  (a) (2 points) For each classifier, you should report the accuracy of the prediction, where accuracy is the percentage of the correctly classified instances. 


# (b) (3 points) For each classifier, please report the errors or costs in each iteration and plot figures for the errors/costs for all the iterations.


# (c) (5 points) Each classifier needs to be tested using two datasets: (1) Iris (by treating one class as positive class and the other two classes as negative class) and (2) another dataset. You need to find your second dataset from UCI machine learning repository. This dataset needs to be bigger than the Iris dataset (more samples and more features).


# (d) (5 points) Properly analyze the classifiers’ behavior. For example, how do your classifiers converge? what is the effect of feature scaling to your classifiers.


# (e) (3 points) Analysis on any other aspects that are not mentioned above and that you think important. For example, the effect of different learning rates on model convergence. 


