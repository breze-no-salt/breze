# -*- coding: utf-8 -*-


import numpy as np

import theano.tensor as T
import theano.tensor.extra_ops
from theano.tensor.shared_randomstreams import RandomStreams


def normal_pdf(x, location=0, scale=1):
    z = 1. / (np.sqrt(2 * np.pi) * scale)
    exp_arg = -((x - location) ** 2) / (2 * scale ** 2)
    return T.exp(exp_arg) * z


def normal_cdf(x, location=0, scale=1):
    erf_arg = (x - location) / T.sqrt(2 * scale ** 2)
    return .5 * (1 + T.erf(erf_arg))


def rectifier(mean, var):
    std = T.sqrt(var)
    ratio = mean / std

    mean_ = normal_cdf(ratio) * mean + normal_pdf(ratio) * std

    A = mean * std * normal_pdf(ratio)
    B = (mean ** 2 + std ** 2) * normal_cdf(ratio)
    exp_of_squared = A + B

    var_ = exp_of_squared - mean_ ** 2
    return mean_, var_


def identity(mean, var):
    return mean, var


def sampling_softmax(axis=1, rng=None):
    if rng is None:
        rng = RandomStreams()

    def inner(mean, var):
        # Generate samples of the distribution.
        samples = rng.normal(size=mean.shape)
        samples = samples * T.sqrt(var) + mean

        # Softmax them.
        # Subtract minimum for numerical stability.
        samples -= samples.min(axis=axis).dimshuffle(0, 'x')
        exped = T.exp(samples)
        normalizer = exped.sum(axis=axis)
        if axis == 1:
            result = exped / normalizer.dimshuffle(0, 'x')
        if axis == 2:
            result = exped / normalizer.dimshuffle(0, 1, 'x')

        return result, T.zeros_like(var)

    return inner
