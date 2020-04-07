# Cyrus Baker
# March 30, 2020
# HW5 - main.py

# Dependencies
import pandas as pd
import numpy as np
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

import preprocess
import dimRed

algo = sys.argv[1]
dataset = sys.argv[2]

if algo not in ["PCA", "LDA", "KernalPCA"]:
    sys.exit("Algorithm not found")

if dataset not in ["iris", "MNIST"]:
    sys.exit("Dataset not found")



