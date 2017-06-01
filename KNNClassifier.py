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

    def fit(self, X, y):

        for k in range(1, self.kmax):
            self.KNNclsf = KNeighborsClassifier(n_neighbors=self.neigbours)
            # TODO probar con weights 'distance' tambien
            scores = cross_val_score(self.KNNclsf, X, y, cv=5, scoring='f1_macro')

    def predict(self):

        return self.KNNclsf.predict

