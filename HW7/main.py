# Cyrus Baker
# April 20, 2020
# HW7 - main.py

# dependencies
import time
import sys

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from xlsxwriter.utility import xl_rowcol_to_cell

import clusterers

dataset = sys.argv[1]
clusterer = sys.argv[2]

if dataset not in ['iris', 'faulty']:
    sys.exit("Dataset not found")

if clusterer not in ['kmeans', 'sci_hei', 'skl_hei', 'dbscan']:
    sys.exit("Clusterer not found")


if dataset == "iris":
    df = pd.read_csv("iris.data", header=None)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    names = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    y = [names[y[i]] for i in range(len(y))]
    k = 3

else:
    df = pd.read_csv("faults.csv")

    targets = df.iloc[:, -7:]
    df.drop(targets.columns, axis=1, inplace=True)
    df['Target'] = targets.idxmax(1)

    df['TypeOfSteel_A300'] = df['TypeOfSteel_A300'].astype(
        'category', copy=False)
    df['TypeOfSteel_A400'] = df['TypeOfSteel_A400'].astype(
        'category', copy=False)
    df['Outside_Global_Index'] = df['Outside_Global_Index'].astype(
        'category', copy=False)
    X = df.loc[:, df.columns != 'Target']
    y = df.loc[:, 'Target']
    names = {"Pastry": 0, "Z_Scratch": 1, "K_Scatch": 2,
             "Stains": 3, "Dirtiness": 4, "Bumps": 5, "Other_Faults": 6}
    y = [names[y[i]] for i in range(len(y))]
    k = 7

# Elbow Method
sse = []

for n in range(1, 20):
    km = KMeans(n_clusters=n,
                init="k-means++",
                max_iter=300,
                random_state=0
                )
    km.fit(X)
    sse.append(km.inertia_)

plt.plot(range(1, 20), sse)
plt.title("Ideal Number of Clusters Using Elbow Method")
plt.ylabel("SSE")
plt.xlabel("Number of clusters")
# plt.show()

# minPts and eps
num = 10
nn = NearestNeighbors(n_neighbors=num + 1)
neighbors = nn.fit(X)
distances, indices = neighbors.kneighbors(X)
distanceK = np.empty([num, X.shape[0]])
for i in range(num):
    di = distances[:, (i+1)]
    di.sort()
    di = di[::-1]
    distanceK[i] = di
for i in range(num):
    plt.plot(distanceK[i], label="K=%d" % (i+1))
    plt.ylabel("Distance")
    plt.xlabel("Points")
    plt.legend()
    # plt.show()

if clusterer == 'kmeans':
    cl = clusterers.Clusterers(n_clusters=k,
                               init="k-means++",
                               max_iter=300,
                               tol=1e-04,
                               affinity="euclidean",
                               linkage="single",
                               method="single",
                               metric="euclidean",
                               t=2.5,
                               criterion="distance",
                               eps=0.2,
                               min_samples=5,
                               random_state=0,
                               X=X
                               )
    y_pred = cl.kmeans()

if clusterer == 'sci_hei':
    cl = clusterers.Clusterers(n_clusters=k,
                               init="k-means++",
                               max_iter=500,
                               tol=1e-04,
                               affinity="euclidean",
                               linkage="single",
                               method="single",
                               metric="euclidean",
                               t=2.5,
                               criterion="distance",
                               eps=0.2,
                               min_samples=5,
                               random_state=0,
                               X=X
                               )
    y_pred = cl.sci_hei()
if clusterer == 'skl_hei':
    cl = clusterers.Clusterers(n_clusters=k,
                               init="k-means++",
                               max_iter=500,
                               tol=1e-04,
                               affinity="euclidean",
                               linkage="single",
                               method="single",
                               metric="euclidean",
                               t=2.5,
                               criterion="distance",
                               eps=0.2,
                               min_samples=5,
                               random_state=0,
                               X=X
                               )
    y_pred = cl.skl_hei()
if clusterer == 'dbscan':
    cl = clusterers.Clusterers(n_clusters=k,
                               init="k-means++",
                               max_iter=500,
                               tol=1e-04,
                               affinity="euclidean",
                               linkage="single",
                               method="single",
                               metric="euclidean",
                               t=2.5,
                               criterion="distance",
                               eps=0.2,
                               min_samples=5,
                               random_state=0,
                               X=X
                               )
    y_pred = cl.dbscan()

c = 0
for i in range(X.shape[0]):
    if y[i] == y_pred[i]:
        c += 1
acc = c / X.shape[0] * 100
print("Accuracy score for ", clusterer, ": ", str(acc))
