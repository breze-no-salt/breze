# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T
from theano.sandbox.linalg.ops import Det, MatrixInverse, psd

det = Det()
minv = MatrixInverse()


# TODO unify with ../distributions.py -- maybe mimic scipy api?


def pdf(sample, mean, cov):
    dim = sample.shape[0]
    psd(cov)
    inv_cov = minv(cov)

    part_func = (2 * np.pi) ** (dim / 2.) * det(cov) ** 0.5

    mean = T.shape_padleft(mean)
    residual = sample - mean
    density = T.exp(-.5 * T.dot(T.dot(residual, inv_cov), residual.T))

    return density / part_func


def logpdf(sample, mean, cov):
    psd(cov)
    inv_cov = minv(cov)

    log_part_func = (
        - .5 * T.log(det(cov))
        - .5 * sample.shape[0] * T.log(2 * np.pi))

    mean = T.shape_padleft(mean)
    residual = sample - mean
    log_density = - .5 * T.dot(T.dot(residual, inv_cov), residual.T)

    return log_density + log_part_func
