from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import math


class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.neigbours = 0
        self.KNNclsf = None

    def fit(self):

        self.KNNclsf = KNeighborsClassifier(n_neighbors=self.neigbours, weights=self.weights)

    def predict(self):

        return self.KNNclsf.predict

