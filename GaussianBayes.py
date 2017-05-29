# Author: Jose' Miguel Buenaposada (josemiguel.buenaposada@urjc.es)

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn import covariance
import numpy as np
import math

class GaussianBayes(BaseEstimator, ClassifierMixin):
    """
    Gaussian Bayes Classifier (GaussianBayes)
    Min error classifier with gaussian classes asumption (aka Gaussian Bayes Classifier)
    Attributes
    ----------
    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.
    mean_ : array, shape (n_classes, n_features)
        mean of each feature per class
    cov_ : array, shape (n_classes, n_features, n_classes)
        covariance matrix of each feature per class
    inv_cov_ : array, shape (n_classes, n_features, n_classes)
        inverse covariance of each feature per class
    class_prior: array, shape (n_classes, 1)
        priors values for each class
    """

    def __init__(self, class_prior = None):
        self.class_count_ = None
        self.mean_ = None
        self.cov_ = None
        self.invcov_ = None
        self.class_prior_ = None


    def fit(self, X, y, fit_prior=True, class_prior=None):
        """Fit Gaussian Bayes according to X, y
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        fit_prior: boolean
            If True the priors are estimated from data. When False
            the priors are in class_prior parameter
        class_prior: array, shape (n_classes, 1)
             Priors values for each class
        Returns
        -------
        self : object
            Returns self.
        """

        # FIXME! The input parameters checking is not implemented!

        n_classes = len(np.unique(y))
        _, n_features = X.shape
        self.mean_ = np.zeros((n_classes, n_features))
        self.cov_ = np.zeros((n_features, n_features, n_classes))
        self.invcov_ = np.zeros((n_features, n_features, n_classes))

        self.class_prior_ = class_prior
        if fit_prior:
           self.class_prior_ = np.zeros((n_classes, 1))
        self.class_count_ = np.zeros((n_classes, 1))

        # We can choose to estimate covariance matrix by the usual sample covariance (np.cov) or
        # use something better (a regularized covariance estimator). Doing something better in the
        # estimation means to take care of possible insufficient amount of data. We choose here to use
        # The Ledoit-Wolf way of estimating covariance estimation (see docs in sklearn).
        CovEstimator = covariance.LedoitWolf(assume_centered=False, store_precision=True) # Regularized covariance estimator
        labels = np.unique(y)
        for i in range(len(labels)):
            c = labels[i]
            self.class_count_[i] = np.sum(y==c)
            Xc = X[(y==c).ravel(), :] # c class data
            self.mean_[i,:] = np.mean(Xc, axis=0)
            CovEstimator.fit(Xc) # store_precision=True to get the inverse covariance too.
            self.cov_[:,:,i] = CovEstimator.covariance_
            self.invcov_[:,:,i] = CovEstimator.precision_ # inverse cov. matrix is call precision
#            self.cov_[:,:,c] = np.cov(Xc, rowvar=0) # data in rows: rowvar = 0
#            self.invcov_[:,:,i] = np.linalg.inv(self.cov_[:,:,i])

        if fit_prior or (class_prior == None) or (class_prior.empty()) or (class_prior.len() != n_classes):
            self.class_prior_ = self.class_count_ / np.sum(self.class_count_)
        else:
            self.class_prior_ = class_prior


        return self


    def predict(self, X):
        """
        Perform classification on an array of test vectors X (samples by rows)
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        posteriors = self.predict_proba(X)

        return np.argmax(posteriors, axis=1)


    def predict_proba(self, X):
        """
        Perform classification on an array of test vectors X (samples by rows)
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        posteriors : array-like, shape = [n_samples, n_classes]
                     probability for each class per sample.
        """

        # FIXME! The input parameters checking is not implemented!

        n_classes = self.mean_.shape[0]
        n_features = self.cov_.shape[0]
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, n_classes))
        for c in range(len(self.class_prior_)):
            invcov_ = self.invcov_[:,:,c]
            diff_i = X - self.mean_[c,:]

            det_ = np.linalg.det(self.cov_[:,:,c])
            constant = 1. / math.sqrt(pow((2.0*math.pi),n_features) * det_)
            # Speed up trick for:
            #for j in xrange(n_samples):
            #    posteriors_g[c,j] = np.dot(diff_i[j,:], np.dot(invcov_,diff_i[j,:].T))
            exp_part_ = np.dot(diff_i, invcov_)
            fdp_aux = np.multiply(exp_part_, diff_i)
            fdp_data = constant * np.exp(-0.5*np.sum(fdp_aux, axis=1))
            posteriors[:, c] += fdp_data * self.class_prior_[c]

        # Sum-to-1 posteriors normalization (sum elements in every row and divide the
        # elements of the row by the sum of its elements).
        inv_sum = 1./np.sum(posteriors, axis=1)
        np.multiply(posteriors, inv_sum[:, np.newaxis])
        posteriors = np.multiply(posteriors, inv_sum[:, np.newaxis])

        return posteriors
