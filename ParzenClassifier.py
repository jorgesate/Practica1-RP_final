from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


class ParzenClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, rmax, rmin=25, rstep=1):
        self.rmax = rmax
        self.rmin = rmin
        self.rstep = rstep
        self.rs = np.arange(self.rmin, self.rmax + 1, self.rstep)
        self.PARZclsf_best = None
        self.r_best = 0

    def fit(self, X, y):

        mean = []
        for r in self.rs:
            PARZclsf = RadiusNeighborsClassifier(radius=r, weights='distance')
            # TODO probar con weights 'distance' tambien. default: weights='uniform'
            score_r = cross_val_score(PARZclsf, X, y, cv=5, scoring='f1_macro')
            mean.append(np.mean(score_r))

        n_max = np.argmax(mean)
        self.r_best = self.rs[n_max]
        self.PARZclsf_best = RadiusNeighborsClassifier(radius=self.r_best, weights='uniform').fit(X, y)

        return self

    def predict(self, X):

        return self.PARZclsf_best.predict(X)

    '''
    # def fit(self, X, y):
    #
    #     mean = []
    #     for r in self.rs:
    #         PARZclsf = KernelDensity(bandwidth=r)
    #         score_r = cross_val_score(PARZclsf, X, y, cv=5, scoring='f1_macro')
    #         mean.append(np.mean(score_r))
    #
    #     print(mean)
    #     n_max = np.argmax(mean)
    #     self.r_best = self.rs[n_max]
    #     self.PARZclsf_best = KernelDensity(bandwidth=self.r_best).fit(X, y)
    #
    #     return self
    #
    # def predict(self, X):
    #
    #     return np.argmax(self.PARZclsf_best.score_samples(X))
    '''
