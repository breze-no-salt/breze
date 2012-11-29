# -*- coding: utf-8 -*-

"""This module provides functionality for principal component analysis."""


import numpy as np
import scipy.linalg


class Pca(object):

    def __init__(self, n_components=None, whiten=False):
        """Create a Pca object.

        :param n_components: Amount of components to keep.
        :param whiten: Flag that indicates whether the data should have
            isometric and unit covariance after transforming.
        """
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X):
        """Fit the parameters of the model.

        The data should be centered (that is, its mean subtracted rowwise)
        before using this method.

        :param X: An array of shape `(n, d)` where `n` is the number of
            data points and `d` the input dimensionality."""
        n_components = X.shape[1] if self.n_components is None else self.n_components
        cov = np.cov(X, rowvar=0)
        w, s, v = scipy.linalg.svd(cov, full_matrices=False)
        w = w[:, :n_components]
        if self.whiten:
            s_ = s[:n_components]
            w = np.dot(w, np.diag(1. / np.sqrt(s_)))

        self.weights = w
        self.singular_values = s

    def transform(self, X):
        """Transform data according to the model.

        :param X: An array of shape `(n, d)` where `n` is the number of
            data points and `d` the input dimensionality.
        :returns: An array of shape `(n, c)` where `n` is the number of samples
            and `c` is the number of components kept."""
        return np.dot(X, self.weights)
