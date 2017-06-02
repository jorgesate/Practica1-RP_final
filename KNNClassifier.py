from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import math

# KNeighborsClassifier(n_neighbors=5, weights='uniform''distance', algorithm='auto',
#       leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)

FITERROR = "KNNClassifier is not fited. Need to fit the estimator first!"


class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, kmax=50, kmin=1, kstep=1):
        self.neigbours = 0
        self.kmax = kmax
        self.kmin = kmin
        self.kstep = kstep
        self.ks = np.arange(self.kmin, self.kmax, self.kstep)
        self.KNNclsf_best = None
        self.scores = np.zeros_like(self.ks)


    def fit(self, X, y):

        for k in enumerate(self.ks):
            KNNclsf = KNeighborsClassifier(n_neighbors=k, weights='uniform')
            # TODO probar con weights 'distance' tambien
            score_k = cross_val_score(KNNclsf, X, y, cv=5, scoring='f1_macro')
            self.scores[k] = score_k.mean()

        n_max = np.argmax(self.scores)
        k_max = self.scores[n_max]
        self.KNNclsf_best = KNeighborsClassifier(n_neighbors=k_max).fit(X, y)

        return self

    def predict(self, X):

        if self.KNNclsf_best is not None:
            return self.KNNclsf_best.predict(X)
        else:
            raise AttributeError(FITERROR)

