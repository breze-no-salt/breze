# -*- coding: utf-8 -*-

"""Module holding miscellaneous functionality."""


import theano.tensor as T
import warnings
import norm as norm_
from ..util import lookup


def pairwise_diff(X, Y=None):
    """Given two arrays with samples in the row, compute the pairwise
    differences.

    Parameters
    ----------

    X : Theano variable
        Has shape ``(n, d)``. Contains one item per first dimension.

    Y : Theano variable, optional [default: None]
        Has shape ``(m, d)``.  If not given, defaults to ``X``.

    Returns
    -------

    res : Theano variable
        Has shape ``(n, d, m)``.
    """
    Y = X if Y is None else Y
    diffs = X.T.dimshuffle(1, 0, 'x') - Y.T.dimshuffle('x', 0, 1)
    return diffs


def distance_matrix(X, Y=None, norm=norm_.l2):
    """Return an expression containing the distances given the norm of up to two
    arrays containing samples.

    Parameters
    ----------

    X : Theano variable
        Has shape ``(n, d)``. Contains one item per first dimension.

    Y : Theano variable, optional [default: None]
        Has shape ``(m, d)``.  If not given, defaults to ``X``.

    norm : string or callable
        Either a string pointing at a function in ``breze.arch.component.norm``
        or a function that has the same signature as these.


    Returns
    -------

    res : Theano variable
        Has shape ``(n, m)``.
    """
    diff = pairwise_diff(X, Y)
    return distance_matrix_by_diff(diff, norm=norm)


def distance_matrix_by_diff(diff, norm=norm_.l2):
    """Return an expression containing the distances given the norm ``norm``
    arrays containing samples.

    Parameters
    ----------

    D : Theano variable
        Has shape ``(n, d, m)`` and represents differences between two
        collections of the same set.

    norm : string or callable
        Either a string pointing at a function in ``breze.arch.component.norm``
        or a function that has the same signature as these.


    Returns
    -------

    res : Theano variable
        Has shape ``(n, m)``.
    """
    if isinstance(norm, (str, unicode)):
        norm = lookup(norm, norm_)
    dist_comps = norm(diff, axis=1)
    return dist_comps


def cat_entropy(arr):
    """Return the entropy of categorical distributions described by the rows
    in ``arr``.

    Parameters
    ----------

    arr : Theano variable
        Array of shape ``(n, d)`` describing ``n`` different categorical
        variables. Rows need to sum up to ``1`` and be non-negative.

    Returns
    -------

    res : theano variable
        Has shape ``(n,)``.
    """
    # TODO check if this is also valid for multinomial.
    arr = T.minimum(1, arr + 1e-8)
    return -(arr * T.log(arr)).sum(axis=1)


def project_into_l2_ball(arr, radius=1):
    """Return ``arr`` projected into the L2 ball.

    Parameters
    ----------

    arr : Theano variable
        Array of shape either ``(n, d)`` or ``(d,)``. If the former, all rows
        are projected individually.

    radius : float, optional [default: 1]


    Returns
    -------

    res : Theano variable
        Projected result of the same shape as ``arr``.
    """
    # Distinguish whether we are given a single or many vectors to work upon.
    batch = arr.ndim == 2
    if not batch:
        arr = T.shape_padleft(arr)

    lengths = T.sqrt((arr ** 2).sum(axis=1)).dimshuffle(0, 'x')
    arr = T.switch(lengths > T.sqrt(radius), arr / lengths * radius, arr)

    if not batch:
        arr = arr[0]

    return arr

def inter_laplace_kl(mean, b, mean_=0, b_=1, b_offset=0, b_offset_=0):
    """Function returning a theano tensor representing the Kullback-Leibler
    divergence between Laplace distributed random variables and a Laplace
    with zero mean and b of one.

    Parameters
    ----------

    mean : Theano variable
        Representation of the mean of the input.

    var : Theano variable
        Representation of the scale b of the input. Has to have the same shape
        as ``mean``. Needs to be positive.


    Returns
    -------

    kl : Theano variable
        Same shape as ``mean`` and ``var``. Each point represents the KL
        divergence between the a standard and diag Laplace given by
        ``mean`` and ``var``.
    """
    m1, b1, m2, b2 = mean, b + b_offset, mean_, b_ + b_offset_
    if b2 == 0 and m2 == 0:
        # return 1.0 - b1*T.exp(-abs(m1)/(b1 + 1e-4)) - abs(m1) + T.log(b1 + 1e-4)
        return 1.0 + T.log((b1 + 1e-4)) - abs(m1) - b1*T.exp(-abs(m1)/(b1 + 1e-4))
    else:
        warnings.warn("Warning: untested implemenation under these parameter settings")
        return b1/b2 * (abs(m1-m2)/b1 + T.exp(-abs(m1-m2)/b1)) + T.log(b2/b1) - 1

def inter_gauss_kl(mean, var, mean_=0, var_=1, var_offset=0, var_offset_=0):
    """Function returning a theano tensor representing the Kullback-Leibler
    divergence between Gaussian distributed random variables and a white
    Gaussian.

    Parameters
    ----------

    mean : Theano variable
        Representation of the mean of the input.

    var : Theano variable
        Representation of the variance of the input. Has to have the same shape
        as ``mean``. Needs to be positive.


    Returns
    -------

    kl : Theano variable
        Same shape as ``mean`` and ``var``. Each point represents the KL
        divergence between the a standard normal and the Gaussian given by
        ``mean`` and ``var``.
    """
    #return -.5 * (1 + T.log(var + 1e-8) - mean ** 2 - var)
    #std = T.sqrt(var)
    #std_ = T.sqrt(var_)
    m1, s1, m2, s2 = mean, T.sqrt(var + var_offset), mean_, T.sqrt(var_ + var_offset_)
    return T.log(s2 / s1 + 1e-4) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2 + 1e-4) - .5

