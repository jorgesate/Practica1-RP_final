from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# KNeighborsClassifier(n_neighbors=5, weights='uniform''distance', algorithm='auto',
#       leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)


class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, kmax, kmin=1, kstep=1):
        self.kmax = kmax
        self.kmin = kmin
        self.kstep = kstep
        self.ks = np.arange(self.kmin, self.kmax + 1, self.kstep)
        self.KNNclsf_best = None
        self.k_best = 0
        self.weights = ['uniform', 'distance']
        self.w_best = None

    def fit(self, X, y):

        mean_w = []
        for w in self.weights:

            mean_k = []
            for k in self.ks:
                KNNclsf = KNeighborsClassifier(n_neighbors=k, weights=w)
                # TODO probar con weights 'distance' tambien. default: weights='uniform'
                score_k = cross_val_score(KNNclsf, X, y, cv=5, scoring='f1_macro')
                mean_k.append(np.mean(score_k))

            m_max = np.argmax(mean_k)
            self.k_best = self.ks[m_max]
            mean_w.append(self.k_best)

        w_max = np.argmax(mean_w)
        self.w_best = self.weights[w_max]

        self.KNNclsf_best = KNeighborsClassifier(n_neighbors=self.k_best, weights=self.w_best).fit(X, y)

        return self

    def predict(self, X):

        return self.KNNclsf_best.predict(X)



