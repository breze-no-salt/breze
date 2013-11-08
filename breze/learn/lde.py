#!/usr/bin/env python
"""Module for the linear denoiser."""

__authors__ = ('Christian Osendorfer, osendorf@gmail.com',
               'Justin Bayer, bayer.justin@gmail.com')

import numpy as np


from scipy.linalg import lstsq


class LinearDenoiser(object):
    """Class that represents linear denoisers.

    LinearDenoisers (LDEs) were later also named Marginalized Denoising
    AutoEncoders.

    Introduced in [1]_.

    References
    ----------

    .. [1] Xu, Zhixiang Eddie, Kilian Q. Weinberger, and Fei Sha.
           "Rapid feature learning with stacked linear denoisers."
           arXiv preprint arXiv:1105.0972 (2011).
    """

    def __init__(self, p_dropout):
        """Create a LinearDenoiser object.

        Parameters
        ----------

        p_dropout : float
            Probability of an input being dropped out.
        """
        self.p_dropout = p_dropout

    def fit(self, X):
        """Fit the parameters of the model.

        Parameters
        ----------

        X : array_like
            An array of shape ``(n, d)`` where ``n`` is the number of data
            points and ``d`` the input dimensionality."""
        # Add another feature for the bias.
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        n, d = X.shape

        q = np.ones((1, d)) * (1 - self.p_dropout)
        q[0, 0] = 1

        scatter = np.dot(X.T, X)
        Q = scatter * np.dot(q.T, q)
        np.fill_diagonal(Q, q * np.diag(scatter))
        P = scatter * q
        # First row of wm has bias values.
        wm = lstsq((Q + 1e-5 * np.eye(d)).T, P.T[:, 1:])[0]
        self.bias = wm[0]
        self.weights = wm[1:]

    def transform(self, X):
        """Transform data according to the model.

        Parameters
        ----------

        X : array_like
            An array of shape ``(n, d)`` where ``n`` is the number of data
            points and ``d`` the input dimensionality.

        Returns
        -------

        Y : array_like
            An array of shape ``(n, d)`` where ``n`` is the number of data
            points and ``d`` the input dimensionality."""
        return np.dot(X, self.weights) + self.bias
