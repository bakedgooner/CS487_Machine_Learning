# Cyrus Baker
# February 3, 2020
# HW2.py

# Dependencies
import numpy as np
import pandas as pd
import sys

from adaline import Adaline
from perceptron import Perceptron
from sgd import SGD

# (20 points) Write program to test the different classifiers.
# get file and classifier from command line
classifiers  = ['perceptron', 'adaline', 'sgd']
classifier = sys.argv[1]
if (classifiers.index(classifier) == -1):
    print('invalid classifier')
    sys.exit(1)
if (classifier == 'perceptron'):
    cf = Perceptron(learning_rate=0.1, num_iter=50)
elif (classifier == 'adaline'):
    cf = Adaline(learning_rate=0.1, num_iter=50)
elif (classifier == 'sgd'):
    cf = SGD(learning_rate=0.1, num_iter=50)

file_name = sys.argv[2]

# convert the file to dataframe
df = pd.read_csv(file_name, header=None)

def main():
    cf.fit(X, y)
    cf.plot(range(1, len(cf.errors_) + 1), cf.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Updates')
    plt.show()

if __name__ == "__main__":
    main()



# (d) (15 points) It calls the different classifiers to train models, make predictions, and report prediction errors.


# 5. (18 points) Write a report report.pdf to analyze the predictive power and the running time of different classifiers.

#  (a) (2 points) For each classifier, you should report the accuracy of the prediction, where accuracy is the percentage of the correctly classified instances. 


# (b) (3 points) For each classifier, please report the errors or costs in each iteration and plot figures for the errors/costs for all the iterations.


# (c) (5 points) Each classifier needs to be tested using two datasets: (1) Iris (by treating one class as positive class and the other two classes as negative class) and (2) another dataset. You need to find your second dataset from UCI machine learning repository. This dataset needs to be bigger than the Iris dataset (more samples and more features).


# (d) (5 points) Properly analyze the classifiers’ behavior. For example, how do your classifiers converge? what is the effect of feature scaling to your classifiers.


# (e) (3 points) Analysis on any other aspects that are not mentioned above and that you think important. For example, the effect of different learning rates on model convergence. 


