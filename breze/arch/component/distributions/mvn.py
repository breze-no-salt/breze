# -*- coding: utf-8 -*-

"""Module containing expression buildes for the multivariate normal."""


import numpy as np
import theano.tensor as T
from theano.sandbox.linalg.ops import Det, MatrixInverse, psd, Cholesky

det = Det()
minv = MatrixInverse()
chol = Cholesky()


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


    Examples
    --------

    >>> import theano
    >>> import theano.tensor as T
    >>> import numpy as np
    >>> from breze.learn.utils import theano_floatx
    >>> sample = T.matrix('sample')
    >>> mean = T.vector('mean')
    >>> cov = T.matrix('cov')
    >>> p = pdf(sample, mean, cov)
    >>> f_p = theano.function([sample, mean, cov], p)

    >>> mu = np.array([-1, 1])
    >>> sigma = np.array([[.9, .4], [.4, .3]])
    >>> X = np.array([[-1, 1], [1, -1]])
    >>> mu, sigma, X = theano_floatx(mu, sigma, X)
    >>> ps = f_p(X, mu, sigma)
    >>> np.allclose(ps, [4.798702e-01, 7.73744047e-17])
    True
    """

    dim = sample.shape[0]
    psd(cov)
    inv_cov = minv(cov)
    L = chol(inv_cov)

    part_func = (2 * np.pi) ** (dim / 2.) * det(cov) ** 0.5

    mean = T.shape_padleft(mean)
    residual = sample - mean
    B = T.dot(residual, L)
    A = (B ** 2).sum(axis=1)
    density = T.exp(-.5 * A)

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


    Returns
    -------

    l : Theano variable
        Array of shape ``(n,)`` where each entry represents the log density of
        the corresponding sample.


    Examples
    --------

    >>> import theano
    >>> import theano.tensor as T
    >>> import numpy as np
    >>> from breze.learn.utils import theano_floatx
    >>> sample = T.matrix('sample')
    >>> mean = T.vector('mean')
    >>> cov = T.matrix('cov')
    >>> p = logpdf(sample, mean, cov)
    >>> f_p = theano.function([sample, mean, cov], p)

    >>> mu = np.array([-1, 1])
    >>> sigma = np.array([[.9, .4], [.4, .3]])
    >>> X = np.array([[-1, 1], [1, -1]])
    >>> mu, sigma, X = theano_floatx(mu, sigma, X)
    >>> ps = f_p(X, mu, sigma)
    >>> np.allclose(ps, np.log([4.798702e-01, 7.73744047e-17]))
    True
    """
    psd(cov)
    inv_cov = minv(cov)

    inv_cov = minv(cov)
    L = chol(inv_cov)

    log_part_func = (
        - .5 * T.log(det(cov))
        - .5 * sample.shape[1] * T.log(2 * np.pi))

    mean = T.shape_padleft(mean)
    residual = sample - mean
    B = T.dot(residual, L)
    A = (B ** 2).sum(axis=1)
    log_density = - .5 * A

    return log_density + log_part_func
