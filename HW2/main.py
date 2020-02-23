# Cyrus Baker
# February 3, 2020
# main.py

# Dependencies
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from adaline import Adaline
from perceptron import Perceptron
from sgd import SGD

def main():
<<<<<<< Updated upstream:HW2/main.py
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



=======
    # check that all args are valid
    if(len(sys.argv) != 3):
        err = "Invalid number of arguments. "
        print(err + 'Refer to README for more information')
        sys.exit()
    else:
        try:
            df = pd.read_csv(sys.argv[2], header=None)
        except:
            err = 'File ' + sys.argv[2] + ' could not be found. '
            print(err + 'Refer to README for more information')
            sys.exit()
>>>>>>> Stashed changes:HW2/HW2.py
    
    if(sys.argv[1] == 'perceptron'):
        cf = Perceptron()
    elif(sys.argv[1] == 'adaline'):
        cf = Adaline()
    elif(sys.argv[1] == 'sgd'):
        cf = SGD()
    else:
        err = 'Invalid Classifier. '
        print(err + 'Refer to README for more information')
        sys.exit()

    # split into data and target
    X = df.iloc[:,:-1].values
    y = df.iloc[:, -1].values

    
    # change targets to positive and negative
    targs = np.unique(y)
    print(targs)
    y = np.where(y == targs[0], 1, -1)

    # standardize attributes
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)

    # fit dataframe to classifier
    cf.fit(X_std, y)

if __name__ == "__main__":
    main()