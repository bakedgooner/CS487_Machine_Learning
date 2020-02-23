# Cyrus Baker
# February 3, 2020
# HW2.py

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