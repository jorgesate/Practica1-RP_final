from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
import numpy as np


class GMMClassifierSk(BaseEstimator, ClassifierMixin):

    def __init__(self, nmax, nmin=1, nstep=1):
        self.nmax = nmax
        self.nmin = nmin
        self.nstep = nstep
        self.ns = np.arange(self.nmin, self.nmax + 1, self.nstep)
        self.GMMclsf_best = None
        self.n_best = 0
        self.covariances = ['full', 'tied', 'diag', 'spherical']

    def fit(self, X, y):

        for c in self.covariances:
            print(c)

        mean = []
        for n in self.ns:
            GMMclsf = GaussianMixture(n_components=n, covariance_type='full')
            # TODO probar con weights 'distance' tambien. default: weights='uniform'
            score_k = cross_val_score(GMMclsf, X, y, cv=5, scoring='f1_macro')
            mean.append(np.mean(score_k))

        n_max = np.argmax(mean)
        self.n_best = self.ns[n_max]
        self.GMMclsf_best = GaussianMixture(n_components=n, covariance_type='full').fit(X, y)

        return self

    def predict(self, X):

        return self.GMMclsf_best.predict(X)



