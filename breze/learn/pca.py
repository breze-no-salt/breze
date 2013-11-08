# -*- coding: utf-8 -*-

"""This module provides functionality for principal component analysis."""


import numpy as np
import scipy.linalg


class BaseCa(object):

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
            An array of shape ``(n, c)`` where ``n`` is the number of samples
            and ``c`` is the number of components kept."""
        return np.dot(X, self.weights)

    def inverse_transform(self, F):
        """Perform an inverse transformation of transformed data according to
        the model.


        Parameters
        ----------

        F : array_like
            An array of shape ``(n, d)`` where ``n`` is the number of data
            points and ``d`` the dimensionality if the feature space.


        Returns
        -------

        X : array_like
            An array of shape ``(n, c)`` where ``n`` is the number of samples
            and ``c`` is the dimensionality of the input space."""
        return np.dot(F, self.weights.T)

    def reconstruct(self, X):
        """Reconstruct the data according to the model.

        Parameters
        ----------

        X : array_like
            An array of shape ``(n, d)`` where ``n`` is the number of data
            points and ``d`` the input dimensionality.


        Returns
        -------

        Y : array_like
            An array of shape ``(n, d)`` where ``n`` is the number of samples
            and ``d`` is the dimensionality of the input space."""
        F = self.transform(X)
        return self.inverse_transform(F)


class Pca(BaseCa):
    """Class to perform principal component analysis.

    Attributes
    ----------

    n_components : integer
        Number of components to keep.

    whiten : boolean
        Flag indicating whether to whiten the covariance matrix.

    weights : array_like
        2D array representing the map from observable to latent space.

    singular_values : array_like
        1D array containing the singular values of the problem.
    """

    def __init__(self, n_components=None, whiten=False):
        """Create a Pca object.

        Paramters
        ---------

        n_components : integer
            Amount of components to keep.
        whiten : boolean
            Flag that indicates whether the data should have isometric and unit
            covariance after transforming.
        """
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X):
        """Fit the parameters of the model.

        The data should be centered (that is, its mean subtracted rowwise)
        before using this method.

        Parameters
        ----------

        X : array_like
            An array of shape ``(n, d)`` where ``n`` is the number of data
            points and ``d`` the input dimensionality."""
        n_components = X.shape[1] if self.n_components is None else self.n_components
        cov = np.cov(X, rowvar=0)
        w, s, v = scipy.linalg.svd(cov, full_matrices=False)
        w = w[:, :n_components]
        if self.whiten:
            s_ = s[:n_components]
            w = np.dot(w, np.diag(1. / np.sqrt(s_)))

        self.weights = w
        self.singular_values = s


class Zca(BaseCa):
    """Class to perform zero component analysis.

    Attributes
    ----------

    min_eig_val : float
        Eigenvalues are increased by this value before reconstructing.

    weights : array_like
        2D array representing the map from observable to latent space.

    singular_values : array_like
        1D array containing the singular values of the problem.
    """

    def __init__(self, min_eig_val=0.1):
        """Create a Zca object.

        Paramters
        ---------

        min_eig_val : float, optional [defalt: 0.1]
            Eigenvalues are increased by this value before reconstructing.
        """
        self.min_eig_val = min_eig_val

    def fit(self, X):
        """Fit the parameters of the model.

        The data should be centered (that is, its mean subtracted rowwise)
        before using this method.

        Paramters
        ---------

        X : array_like
            An array of shape ``(n, d)`` where ``n`` is the number of data
            points and ``d`` the input dimensionality."""
        cov = np.cov(X, rowvar=0)
        w, s, v = scipy.linalg.svd(cov, full_matrices=False)
        w = np.dot(np.dot(w, np.diag(1. / np.sqrt(s + self.min_eig_val))), w.T)

        self.weights = w
        self.singular_values = s
