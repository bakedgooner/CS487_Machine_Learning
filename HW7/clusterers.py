# Cyrus Baker
# April 20, 2020
# HW7 - clusterers.py

# dependencies
import time

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


class Clusterers(object):
    def __init__(self,
                 n_clusters,
                 init,
                 max_iter,
                 tol,
                 affinity,
                 linkage,
                 method,
                 metric,
                 t,
                 criterion,
                 eps,
                 min_samples,
                 random_state,
                 X=[]
                 ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.affinity = affinity
        self.linkage = linkage
        self.method = method
        self.metric = metric
        self.t = t
        self.criterion = criterion
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.X = X
        self.__obj = None

    def __fit_predict(self):
        start = time.time() * 1000
        y = self.__obj.fit_predict(self.X)
        end = time.time() * 1000
        print("Running time for fit_predict: ", end - start, " ms")
        return y

    def kmeans(self):
        self.__obj = KMeans(n_clusters=self.n_clusters,
                            init=self.init,
                            max_iter=self.max_iter,
                            tol=self.tol,
                            random_state=self.random_state,
                            )
        return self.__fit_predict()

    def sci_hei(self):
        start = time.time() * 1000
        row = linkage(y=self.X,
                      method=self.method,
                      metric=self.metric
                      )
        end = time.time() * 1000
        print("Running time for sci_hei: ", end - start, " ms")
        return fcluster(row,
                        t=self.t,
                        criterion=self.criterion
                        )

    def skl_hei(self):
        self.__obj = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             affinity=self.affinity,
                                             linkage=self.linkage
                                             )
        return self.__fit_predict()

    def dbscan(self):
        self.__obj = DBSCAN(eps=self.eps,
                            min_samples=self.min_samples,
                            metric=self.metric
                            )
        return self.__fit_predict()
