# Author: Jose' Miguel Buenaposada (josemiguel.buenaposada@urjc.es)

import numpy as np
import math
from sklearn import covariance
from sklearn.mixture import GaussianMixture
from GaussianBayes import *
from sklearn.model_selection import cross_val_score


#An example of a class
class GMMBayes:
    """
    Gaussian Mixture Model Bayes Classifier (GaussianBayes)
    Min error classifier with mixtures of gaussians classes asumption (aka GMM Bayes Classifier)
    Attributes
    ----------
    classes_ is a list of dictionaries, each class is a dictionary
             with three fields:
                - 'covs', a numpy array (n_features x n_features x n_gaussians)
                - 'means', a numpy array (n_gaussians x n_features)
                - 'weights', a numpy array (n_gaussians x 1)
                - 'prior', a float with class prior.
    """

    def __init__(self, classes):
        # classes is a list of dictionaries, each class is a dictionary
        # with three fields:
        #   - 'covs', a numpy array (n_features x n_features x n_gaussians)
        #   - 'means', a numpy array (n_gaussians x n_features)
        #   - 'weights', a numpy array (n_gaussians x 1)
        #   - 'prior', a float with class prior.

        self.classes_ = classes
        self.mean_ = 0
        self.cov_ = 0
        self.invcov_ = 0
        self.class_prior_ = None
        self.class_count_ = None


    def fit(self, X, y, fit_prior=True, class_prior=None):
        """IT DOES NOTHING BY NOW TO Fit Gaussian Bayes according to X, y
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        Returns
        -------
        self : object
            Returns self.
        """

        mean_w = []
        for w in self.weights:

            mean_k = []
            for k in self.ks:
                gaussian_clsf = GaussianBayes()
                score_k = cross_val_score(gaussian_clsf, X, y, cv=5, scoring='f1_macro')
                mean_k.append(np.mean(score_k))

            m_max = np.argmax(mean_k)
            self.k_best = self.ks[m_max]
            mean_w.append(max(mean_k))

        w_max = np.argmax(mean_w)
        self.w_best = self.weights[w_max]

        self.KNNclsf_best = GaussianBayes().fit(X, y)

        return self


    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        n_classes = len(self.classes_) # 4
        n_features = self.classes_[0]['covs'].shape[0] # 2
        n_samples = X.shape[0]
        posteriors = np.zeros((n_classes,n_samples))
        for c in range(n_classes):
            class_c = self.classes_[c]
            n_gaussians = len(class_c['weights'])
            means_ = class_c['means']
            covs_ = class_c['covs']
            weights_ = class_c['weights']
            prior_ = class_c['prior']
            for g in range(n_gaussians):
                if n_gaussians > 1:
                    mean_ = means_[g,:]
                    cov_  = covs_[:,:,g]
                else:
                    mean_ = means_
                    cov_  = covs_
                invcov_ = np.linalg.inv(cov_)
                weight_ = weights_[g]
                det_ = np.linalg.det(cov_)
                diff_i = X - mean_
                constant = 1. / math.sqrt(pow((2.0*math.pi),n_features) * det_)

                # Speed up trick for:
                #for j in xrange(n_samples):
                #    posteriors_g[c,j] = np.dot(diff_i[j,:], np.dot(invcov_,diff_i[j,:].T))
                exp_part_ = np.dot(diff_i, invcov_)
                fdp_aux = np.multiply(exp_part_, diff_i)
                fdp_data = constant * np.exp(-0.5*np.sum(fdp_aux, axis=1)) * weight_
                posteriors[c,:] += fdp_data * prior_

        return np.argmax(posteriors, axis=0)

