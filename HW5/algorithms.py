# Cyrus Baker
# March 30, 2020
# HW5 - algorithms.py

# dependencies
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.tree import DecisionTreeClassifier
import time


class Algorithms(object):
    def __init__(self, n_components, kernel, C, gamma, max_depth, criterion, random_state, X_train=[], y_train=[], X_test=[]):
        self.n_components = n_components
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.__obj = None

    def __fit(self):
        start = time.time() * 1000
        self.__obj.fit(self.X_train, self.y_train)
        end = time.time() * 1000
        score = self.__obj.score(self.X_train, self.y_train)
        runtime = end - start
        print("\nAccuracy for train: ", score)
        print("Running time for fit: ", runtime, " ms")

    def __predict(self):
        start = time.time() * 1000
        y_pred = self.__obj.predict(self.X_test)
        end = time.time() * 1000
        print("Running time for predict: ", end - start, " ms")
        return y_pred

    def __transform(self):
        start = time.time() * 1000
        X_tr_test = self.__obj.transform(self.X_test)
        end = time.time() * 1000
        print("Running time for transform: ", end - start)
        return X_tr_test

    def __fit_transform(self):
        start = time.time() * 1000
        X_train_red = self.__obj.fit_transform(self.X_train, y=self.y_train)
        end = time.time() * 1000
        print("Running time for fit_transfrom: ", end - start)
        return X_train_red

    def run_pca(self):
        self.__obj = PCA(n_components=self.n_components)
        X_train_pca = self.__fit_transform()
        X_test_pca = self.__transform()
        return X_test_pca, X_train_pca

    def run_lda(self):
        self.__obj = LinearDiscriminantAnalysis(n_components=self.n_components)
        X_train_lda = self.__fit_transform()
        X_test_lda = self.__transform()
        return X_test_lda, X_train_lda

    def run_kernel_pca(self):
        self.__obj = KernelPCA(
            n_components=self.n_components, kernel=self.kernel, gamma=self.gamma)
        X_train_kpca = self.__fit_transform()
        X_test_kpca = self.__transform()
        return X_test_kpca, X_train_kpca

    def decisiontree(self):
        self.__obj = DecisionTreeClassifier(criterion=self.criterion,
                                            max_depth=self.max_depth)
        self.__fit()
        return self.__predict()
