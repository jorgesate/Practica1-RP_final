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
        self.covs = ['full', 'tied', 'diag', 'spherical']
        self.cov_best = None

    def fit(self, X, y):

        mean_covs = []
        for c in self.covs:

            mean_n = []
            for n in self.ns:
                GMMclsf = GaussianMixture(n_components=n, covariance_type=c)
                score_k = cross_val_score(GMMclsf, X, y, cv=5, scoring='f1_macro')
                mean_n.append(np.mean(score_k))

            n_max = np.argmax(mean_n)
            self.n_best = self.ns[n_max]
            mean_covs.append(max(mean_n))

        cov_max = np.argmax(mean_covs)
        self.cov_best = self.covs[cov_max]

        self.GMMclsf_best = GaussianMixture(n_components=self.n_best, covariance_type=self.cov_best).fit(X, y)

        return self

    def predict(self, X):

        return self.GMMclsf_best.predict(X)



