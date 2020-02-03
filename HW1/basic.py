# Cyrus Baker
# basic.py
# January 29, 2020
# Purpose - An introduction to the basics of manipulating datasets in Python

import pandas as pd
import sys
import matplotlib.pyplot as plt

# get file from command line
file_name = sys.argv[1]

# convert the file to dataframe
df = pd.read_csv(file_name, header=None)

# add header row 
df.columns=['sepal length','sepal width','petal length','petal width','class']

# subset of data frame used for problem 4
dfSetosa = df.loc[df['class'] == 'Iris-setosa']

# helper functions
def numRowsCols(data):
    print ("(Rows, Cols) => ", data.shape)
    
def printDistinctTarget(data):
    print (df['class'].unique())

def numRows(data):
    count = data.shape[0]
    return count

def avgColOne(data):
    return data['sepal length'].mean()
    
def maxColTwo(data):
    return data['sepal width'].max()
    
def minColThree(data):
    return data['petal length'].min()

def plotData(data):
    x = data['sepal length']
    y = data['sepal width']
    ratio = x/y
    for name, group in data.groupby("class"):
        plt.scatter(group.index, ratio[group.index], label=name)
    plt.legend()
    plt.xlabel('ID')
    plt.ylabel('Ratio of sepal length to sepal width')
    plt.title('Ratio of length/width and Species')
    plt.show()
    
# results
print('Problem 2: Calculate and print the number of rows and columns that this dataset contains.')
numRowsCols(df)
print("\n")
print('Problem 3: Get all the values of the last column and print the distinct values of the last column.')
printDistinctTarget(df)
print("\n")
print('Problem 4: Calculate the number of rows, the average value of the first column, the maximum value of the second column, and the minimum value of the third column when the last column has value Iris-setosa\n')
print('Number of rows containing Iris-setosa: ', numRows(dfSetosa))
print("\n")
print('Average if first column containing Iris-setosa: ', avgColOne(dfSetosa))
print("\n")
print('Max of second column containing Iris-setosa: ',maxColTwo(dfSetosa))
print("\n")
print('Min of third column containing Iris-setosa: ',minColThree(dfSetosa))
print("\n")
plotData(df)




