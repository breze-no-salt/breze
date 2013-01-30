# -*- coding: utf-8 -*-

"""This module provides functionality for extreme component analysis.

An explanation and derivation of the algorithm can be found in [XCA].

.. [XCA] Extreme component analysis, Welling et al (2003)
"""


import numpy as np
import scipy.linalg


class Xca(object):
    """Class implementing extreme component analysis.

    The idea is that not only the prinicple components or the minor
    components of a data set are important, but a combination of the two. This
    algorithm works by combining probabilistic versions of PCA and MCA.

    The central idea is that if `n` principle and `m` minor components are
    chosen, a gap of size `D - m - n` dimensions is formed in the list of
    singular values. The exact location of this gap is found by chosing the one
    which minimizes a likelihood combining PCA and MCA."""

    def __init__(self, n_components, whiten=False):
        """Create an Xca object.

        :param n_components: Amount of components to keep.
        """
        self.n_components = n_components

    def fit(self, X):
        """Fit the parameters of the model.

        The data should be centered (that is, its mean subtracted rowwise)
        before using this method.

        :param X: An array of shape `(n, d)` where `n` is the number of
            data points and `d` the input dimensionality."""
        n_components = self.n_components
        cov = np.cov(X, rowvar=0)
        w, s, v = scipy.linalg.svd(cov, full_matrices=False)

        _, d = X.shape

        # Find the best gap variable. We will denote the gap variable by k
        # and try all possible positionings.
        eigs = s**2
        sum_eigs = eigs.sum()
        log_eigs = np.log(eigs)
        best_loss, best_k = float('inf'), 0
        for k in range(0, n_components):
            princ_idxs = range(k)
            min_idxs = range(d - k, d)
            idxs = princ_idxs + min_idxs
            loss = (log_eigs[idxs].sum()
                    + (d - n_components) * np.log(sum_eigs - eigs[idxs].sum()))
            if loss < best_loss:
                best_loss = loss
                best_k = k

        w = w[:, range(best_k) + range(d - best_k, d)]

        self.weights = w
        self.singular_values = s
        self.gap_idx = best_k

    def transform(self, X):
        """Transform data according to the model.

        :param X: An array of shape `(n, d)` where `n` is the number of
            data points and `d` the input dimensionality.
        :returns: An array of shape `(n, c)` where `n` is the number of samples
            and `c` is the number of components kept."""
        return np.dot(X, self.weights)

    def inverse_transform(self, F):
        """Perform an inverse transformation of transformed data according to
        the model.

        :param F: An array of shape `(n, d)` where `n` is the number
            of data points and `d` the dimensionality if the feature space.
        :returns: An array of shape `(n, c)` where `n` is the number of samples
            and `c` is the dimensionality of the input space."""
        return np.dot(F, self.weights.T)

    def reconstruct(self, X):
        """Reconstruct the data according to the model.

        :param X: An array of shape `(n, d)` where `n` is the number of
            data points and `d` the input dimensionality.
        :returns: An array of shape `(n, d)` where `n` is the number of samples
            and `d` is the dimensionality of the input space."""
        F = self.transform(X)
        return self.inverse_transform(F)
