# Cyrus Baker
# preprocess.py
# 03/01/2020

#dependencies
import os
import pandas as pd
import numpy as np

def preprocess():
    df = pd.read_csv("eighthr.csv")
    # replace all invalid with NaN
    df.replace(["na", "nan", "NaN", "NaT", "inf", "-inf", "?"], np.nan, inplace=True)
    # drop all rows with NaN
    df = df.dropna()
    # remove date column
    df = df.drop(df.columns[[0]], axis=1)
    return df