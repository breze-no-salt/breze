# -*- coding: utf-8 -*-

"""Slow Feature Analysis.

This module provides functionality for slow feature analysis. A helpful article
is hosted at 
`scholarpedia <http://www.scholarpedia.org/article/Slow_feature_analysis>`_.
"""


import numpy as np
import scipy.linalg


def sfa(X, n_components):
    """Apply slow feature analysis to a dataset and return the corresponding
    weight matrix `w` to extract those components.

    The data set has to be whitened.

    Items can be projected into this space by ``np.dot(X, w)``, where w is the new
    data.

    :param X: List of 2d arrays, where the first dimension indices time and
              the second holds the values of one timestep.
    :param n_components: Amount of components to keep.
    :returns: Frame of the slowest components.
    """
    diff = np.vstack([i[1:] - i[:-1] for i in X])
    cov = scipy.cov(diff, rowvar=0)
    u, s, v = scipy.linalg.svd(cov, full_matrices=False)
    u = u[:, -n_components:][:, ::-1]
    s = s[-n_components:][::-1]
    w = np.dot(u, np.diag(1. / np.sqrt(s)))
    return w
