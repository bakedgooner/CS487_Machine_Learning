# Cyrus Baker
# February 3, 2020
# main.py

# Dependencies
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from adaline import Adaline
from perceptron import Perceptron
from sgd import SGD

classifiers  = ['perceptron', 'adaline', 'sgd']
classifier = sys.argv[1]
if (classifiers.index(classifier) == -1):
    print('invalid classifier')
    sys.exit(1)
if (classifier == 'perceptron'):
    cf = Perceptron(rate=0.1, niter=10)
elif (classifier == 'adaline'):
    cf = Adaline(rate=0.1, niter=10)
elif (classifier == 'sgd'):
    cf = SGD(rate=0.1, niter=10, random_state=1)

file_name = sys.argv[2]

# convert the file to dataframe
df = pd.read_csv(file_name, header=None)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])   
    # plot the decision surface
    x1_min, x1_max = X[:,  0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())  
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
        alpha=0.8, c=cmap(idx),
        marker=markers[idx], label=cl)

def main():
    #Targs = df.iloc[:, -1].unique()
    #if (len(Targs) > 2):
        #Targs = Targs.
        
        
    X = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')

    plt.show()

    if (classifier == 'perceptron'):
        cf.fit(X, y)
        plt.plot(range(1, len(cf.errors) + 1), cf.errors, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')

        plt.show()

        plot_decision_regions(X, y, classifier=cf)
        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')

        plt.show()

    elif (classifier == 'adaline'):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        aln1 = Adaline(0.01, 10).fit(X,y)
        ax[0].plot(range(1, len(aln1.cost) + 1), np.log10(aln1.cost), marker='o')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('log(Sum-squared-error)')
        ax[0].set_title('Adaptive Linear Neuron - Learning rate 0.01')

        aln2 = Adaline(0.0001, 10).fit(X,y)

        ax[1].plot(range(1, len(aln2.cost) + 1), aln2.cost, marker='o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Sum-squared-error')
        ax[1].set_title('Adaptive Linear Neuron - Learning rate 0.0001')
        plt.show()

        aln = Adaline(0.01, 10)
        aln.fit(X_std,y)

        plot_decision_regions(X_std, y, classifier=aln)

        plt.title('Adaptive Linear Neuron - Gradient Descent')
        plt.xlabel('sepal length [standardized]')
        plt.ylabel('petal length [standardized]')
        plt.legend(loc='upper left')
        plt.show()

        plt.plot(range(1, len(aln.cost) + 1), aln.cost, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Sum-squared-error')
        plt.show()
    elif (classifier == 'sgd'):
        cf.fit(X_std, y)
        plot_decision_regions(X_std, y, classifier=cf)
        plt.title('SGD - Stochastic Gradient Descent')
        plt.xlabel('sepal length [standardized]')
        plt.ylabel('petal length [standardized]')
        plt.legend(loc='upper left')
        plt.show()
        plt.plot(range(1, len(cf.cost) + 1), cf.cost, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Average Cost')
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


