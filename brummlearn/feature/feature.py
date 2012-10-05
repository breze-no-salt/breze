# -*- coding: utf-8 -*-

"""This module provides functions for feature engineering."""


import itertools

import np


def rbf(X, n_centers):
    """Return a design matrix with features given by radial basis functions.

    `n_centers` Gaussian kernels are placed along data dimension, equidistant
    between the minimum and the maximum along that dimension. The result then
    contains one column for each of the Kernels.

    :param X: NxD sized array.
    :param n_centers: Amount of Kernels to use for each dimension.
    :returns: Nx(n_centers * D) sized array."""
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    pivots = []
    for i, j in itertools.izip(mn, mx):
        _tmp = np.linspace(i, j, n_centers + 2)
        pivots.append(_tmp[1:-1])
    Y = []
    for row in X:
        _row = []
        for r, cs in itertools.izip(row, pivots):
            width = cs[1] - cs[0]
            for c in cs:
                e = np.exp(-0.5 * ((r - c) / width)**2)
                _row.append(e)
        Y.append(_row)
    return np.asarray(Y)
