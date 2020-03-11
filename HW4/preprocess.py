# Cyrus Baker
# March 9, 2020
# HW4- preprocess.py

#dependencies
import os
import pandas as pd
import numpy as np

def preprocess(fileName):
    df = pd.read_csv(fileName)
    # replace all invalid with NaN
    df.replace(["na", "nan", "NaN", "NaT", "inf", "-inf", "?"], np.nan, inplace=True)
    # drop all rows with NaN
    df = df.dropna()
    # remove date column
    df = df.drop(df.columns[[0]], axis=1)
    return df