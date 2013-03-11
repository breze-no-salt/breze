# -*- coding: utf-8 -*-

"""Module containing kernel functions."""


import theano.tensor as T

from ..component import misc


def linear(X, X_, length_scales, amplitude, diag=False):
    """Return an expression representing a Kernel matrix of the linear kernel
    between rows in ``X`` and rows in ``X_``.

    :parameter X: Array of the size ``(n, d)`` where ``n`` is the number of
        samples and ``d`` is the dimensionality of the data.
    :parameter X_: Array of the size ``(m, d)`` where ``n`` is the number of
        samples and ``d`` is the dimensionality of the data.
    :parameter length_scales: Theano vector representing a parameter for the
        kernel of size ``d``.
    :parameter amplitude: Theano scalar representing the a parameter for the
        overall scale of the kerne.
    :parameter diag: Flag indicating whether the whole Kernel matrix or only
        its diagonal should be computed. If set to ``True``, ``X`` and ``X_``
        have to have the same number of rows.
    :returns: A Theano matrix of size ``(n, m)`` if ``diag`` is ``False``,
        otherwise a Theano vector of size ``n``.
    """
    X = X * length_scales.dimshuffle('x', 0)
    X_ = X_ * length_scales.dimshuffle('x', 0)
    if diag:
        return amplitude * (X * X_).sum(axis=1)
    else:
        return amplitude * T.dot(X, X_.T)


def rbf_by_dist(D2):
    """Return a tensor containing the RBF kernel given a tensor of distances.

    :parameter D2: Tensor of arbitrary size containing Euclidean squared
        distances.
    :returns: Theano tensor of the same size as the input.
    """
    return T.exp(-D2)


def rbf(X, X_, length_scales, amplitude, diag=False, return_distance=False):
    """Return an expression representing a Kernel matrix of the radial basis
    function kernel between rows in ``X`` and rows in ``X_``.

    :parameter X: Array of the size ``(n, d)`` where ``n`` is the number of
        samples and ``d`` is the dimensionality of the data.
    :parameter X_: Array of the size ``(m, d)`` where ``n`` is the number of
        samples and ``d`` is the dimensionality of the data.
    :parameter length_scales: Theano vector representing a parameter for the
        kernel of size ``d``.
    :parameter amplitude: Theano scalar representing the a parameter for the
        overall scale of the kerne.
    :parameter diag: Flag indicating whether the whole Kernel matrix or only
        its diagonal should be computed. If set to ``True``, ``X`` and ``X_``
        have to have the same number of rows.
    :paramter return_distance: If true, a pair is returned of which set the
        second items is the expression of the distance matrix.
    :returns: A Theano matrix of size ``(n, m)`` if ``diag`` is ``False``,
        otherwise a Theano vector of size ``n``.
    """
    X = X * length_scales.dimshuffle('x', 0)
    X_ = X_ * length_scales.dimshuffle('x', 0)
    if not diag:
        D2 = misc.distance_matrix(X, X_, 'l2')
    else:
        D2 = ((X - X_) ** 2).sum(axis=1)

    return amplitude * rbf_by_dist(D2)


def matern52_by_dist(D2):
    """Return a tensor containing the Matern52 kernel given a tensor of
    distances.

    :parameter D2: Tensor of arbitrary size containing Euclidean squared
        distances.
    :returns: Theano tensor of the same size as the input.
    """
    D = T.sqrt(D2 + 1e-8)
    return amplitude * (1.0 + T.sqrt(5.) * D + (5. / 3.) * D2) * T.exp(-T.sqrt(5.) * D)


def matern52(X, X_, length_scales, amplitude, diag=False):
    """Return an expression representing a Kernel matrix of the Matern 5-2
    kernel between rows in ``X`` and rows in ``X_``.

    :parameter X: Array of the size ``(n, d)`` where ``n`` is the number of
        samples and ``d`` is the dimensionality of the data.
    :parameter X_: Array of the size ``(m, d)`` where ``n`` is the number of
        samples and ``d`` is the dimensionality of the data.
    :parameter length_scales: Theano vector representing a parameter for the
        kernel of size ``d``.
    :parameter amplitude: Theano scalar representing the a parameter for the
        overall scale of the kerne.
    :parameter diag: Flag indicating whether the whole Kernel matrix or only
        its diagonal should be computed. If set to ``True``, ``X`` and ``X_``
        have to have the same number of rows.
    :returns: A Theano matrix of size ``(n, m)`` if ``diag`` is ``False``,
        otherwise a Theano vector of size ``n``.
    """
    X = X * length_scales.dimshuffle('x', 0)
    X_ = X_ * length_scales.dimshuffle('x', 0)
    if not diag:
        D2 = misc.distance_matrix(X, X_, 'l2')
    else:
        D2 = ((X - X_) ** 2).sum(axis=1)
    return amplitude * matern52_by_dist(D2)
