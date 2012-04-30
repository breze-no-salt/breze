# -*- coding: utf-8 -*-

"""Principal Component Analysis.

This module provides functionality for principal component analysis."""


import numpy as np
import scipy.linalg


def pca(X, n_components=None, whiten=False):
    """Apply principal component analysis to a dataset and return the 
    corresponding matrix `w` to extract those components.

    The data set has to be whitened.

    Items can be projected into this space by ``np.dot(X, w)``, where w is the
    new frame.

    :param X: 2d array, where the rows index data points and the columns 
              coordinates of those points.
    :param n_components: Amount of components to keep.
    :param whiten: Flag that indicates whether the data should have isometric
                   and unit covariance.
    :returns: ``(w, s)`` where ``w`` is the frame of components with highest 
              variance and ``s`` is the array of eigenvalues which correspond to
              the explained variance.
    """
    if n_components is None:
        n_components = X.shape[1]
    cov = scipy.cov(X, rowvar=0)
    w, s, v = scipy.linalg.svd(cov, full_matrices=False)
    w = w[:, :n_components]
    if whiten:
        s_ = s[:n_components]
        w = np.dot(w, np.diag(1. / np.sqrt(s_)))
    return w, s
