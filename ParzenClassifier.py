from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import cross_val_score
import numpy as np
import math


class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.neigbours = 0
        self.PARZclsf = None

    def fit(self):

        self.PARZclsf = KernelDensity(bandwidth=BW, kernel=self.kernel).fit(X)

    def predict(self):

        return self.PARZclsf.predict

