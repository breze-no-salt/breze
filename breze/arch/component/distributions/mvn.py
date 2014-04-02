# -*- coding: utf-8 -*-

"""Module containing expression buildes for the multivariate normal."""


import numpy as np
import theano.tensor as T
from theano.sandbox.linalg.ops import Det, MatrixInverse, psd

det = Det()
minv = MatrixInverse()


# TODO add examples that also work as tests.


def pdf(sample, mean, cov):
    """Return a theano expression representing the values of the probability
    density function of the multivariate normal.

    Parameters
    ----------

    sample : Theano variable
        Array of shape ``(n, d)`` where ``n`` is the number of samples and
        ``d`` the dimensionality of the data.

    mean : Theano variable
        Array of shape ``(d,)`` representing the mean of the distribution.

    cov : Theano variable
        Array of shape ``(d, d)`` representing the covariance of the
        distribution.


    Returns
    -------

    l : Theano variable
        Array of shape ``(n,)`` where each entry represents the density of the
        corresponding sample.
    """
    dim = sample.shape[0]
    psd(cov)
    inv_cov = minv(cov)

    part_func = (2 * np.pi) ** (dim / 2.) * det(cov) ** 0.5

    mean = T.shape_padleft(mean)
    residual = sample - mean
    density = T.exp(-.5 * T.dot(T.dot(residual, inv_cov), residual.T))

    return density / part_func


def logpdf(sample, mean, cov):
    """Return a theano expression representing the values of the log probability
    density function of the multivariate normal.

    Parameters
    ----------

    sample : Theano variable
        Array of shape ``(n, d)`` where ``n`` is the number of samples and
        ``d`` the dimensionality of the data.

    mean : Theano variable
        Array of shape ``(d,)`` representing the mean of the distribution.

    cov : Theano variable
        Array of shape ``(d, d)`` representing the covariance of the
        distribution.
    """
    psd(cov)
    inv_cov = minv(cov)

    log_part_func = (
        - .5 * T.log(det(cov))
        - .5 * sample.shape[0] * T.log(2 * np.pi))

    mean = T.shape_padleft(mean)
    residual = sample - mean
    log_density = - .5 * T.dot(T.dot(residual, inv_cov), residual.T)

    return log_density + log_part_func
