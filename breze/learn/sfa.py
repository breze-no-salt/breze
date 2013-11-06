# -*- coding: utf-8 -*-

"""Slow Feature Analysis.

This module provides functionality for slow feature analysis. A helpful article
is hosted at
`scholarpedia <http://www.scholarpedia.org/article/Slow_feature_analysis>`_.
"""


import numpy as np
import scipy.linalg


class SlowFeatureAnalysis(object):
    """Class for performing Slow feature analysis.

    Attributes
    ----------

    n_components : integer
        Number of components to keep.
    """

    def __init__(self, n_components=None):
        """Create a SlowFeatureAnalysis object.

        Parameters
        ----------

        n_components : integer
            Amount of components to keep.
        """
        self.n_components = n_components

    def fit(self, X):
        """Fit the parameters of the model.

        The data should be centered (that is, its mean subtracted rowwise)
        and white (e.g. via `pca.Pca`) before using this method.

        Parameters
        ----------

        X : list of array_like
            A list of sequences. Each entry is expected to be an
            array of shape `(*, d)` where `*` is the number of
            data points and may vary from item to item in the list.
            `d` is the input dimensionality and has to be consistent.

        Returns
        -------

        F : list of array_like
            List of sequences. Each item in the list is an array which
            corresponds to the sequence in ``X``. It is of the same shape,
            except that ``d`` is replaced by ``n_components``.
        """
        n_components = X.shape[1] if self.n_components is None else self.n_components
        diff = np.vstack([i[1:] - i[:-1] for i in X])
        cov = scipy.cov(diff, rowvar=0)
        u, _, _ = scipy.linalg.svd(cov, full_matrices=False)
        u = u[:, -n_components:][:, ::-1]

        self.weights = u

    def transform(self, X):
        """Transform data according to the model.

        Parameters
        ----------

        X : array_like
            An array of shape ``(n, d)`` where ``n`` is the number of time
            steps and ``d`` the input dimensionality.

        Returns
        -------

        F : array_like
            An array of shape ``(n, c)`` where ``n`` is the number of time
            steps and ``c`` is the number of components kept."""
        return np.dot(X, self.weights)
