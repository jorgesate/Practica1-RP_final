from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import math

# KNeighborsClassifier(n_neighbors=5, weights='uniform''distance', algorithm='auto',
#       leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)


class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.neigbours = 0
        self.kmax = 5
        self.KNNclsf = None
        self.scores = []

    def fit(self, X, y):

        for k in range(1, self.kmax):
            KNNclsf = KNeighborsClassifier(n_neighbors=k)
            # TODO probar con weights 'distance' tambien
            score_k = cross_val_score(KNNclsf, X, y, cv=5, scoring='f1_macro')
            self.scores[k](score_k.mean())

        ind_max = np.argmax(self.scores)
        self.KNNclsf = KNeighborsClassifier(n_neighbors=ind_max)

    def predict(self, X):

        return self.KNNclsf.predict(X)

