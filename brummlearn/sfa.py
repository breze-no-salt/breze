# -*- coding: utf-8 -*-

"""Slow Feature Analysis.

This module provides functionality for slow feature analysis. A helpful article
is hosted at
`scholarpedia <http://www.scholarpedia.org/article/Slow_feature_analysis>`_.
"""


import numpy as np
import scipy.linalg


class SlowFeatureAnalysis(object):

    def __init__(self, n_components=None):
        """Create a SlowFeatureAnalysis object.

        :param n_components: Amount of components to keep.
        """
        self.n_components = n_components

    def fit(self, X):
        """Fit the parameters of the model.

        The data should be centered (that is, its mean subtracted rowwise)
        and white (e.g. via `pca.Pca`) before using this method.

        :param X: A list of sequences. Each entry is expected to be an
            array of shape `(*, d)` where `*` is the number of
            data points and may vary from item to item in the list.
            `d` is the input dimensionality and has to be consistent."""
        n_components = X.shape[1] if self.n_components is None else self.n_components
        diff = np.vstack([i[1:] - i[:-1] for i in X])
        cov = scipy.cov(diff, rowvar=0)
        u, _, _ = scipy.linalg.svd(cov, full_matrices=False)
        u = u[:, -n_components:][:, ::-1]

        self.weights = u

    def transform(self, X):
        """Transform data according to the model.

        :param X: An array of shape `(n, d)` where `n` is the number of
            time steps and `d` the input dimensionality.
        :returns: An array of shape `(n, c)` where `n` is the number of time
            steps and `c` is the number of components kept."""
        return np.dot(X, self.weights)
