# -*- coding: utf-8 -*-


import numpy as np

import theano.tensor as T
import theano.tensor.extra_ops
from theano.tensor.shared_randomstreams import RandomStreams


epsilon = 1e-4


def normal_pdf(x, location=0, scale=1):
    z = 1. / (np.sqrt(2 * np.pi) * scale + epsilon)
    exp_arg = -((x - location) ** 2) / (2 * scale ** 2 + epsilon)
    return T.exp(exp_arg) * z


def normal_cdf(x, location=0, scale=1):
    erf_arg = (x - location) / T.sqrt(2 * scale ** 2 + epsilon)
    return .5 * (1 + T.erf(erf_arg + epsilon))


def rectifier(mean, var):
    std = T.sqrt(var)
    ratio = mean / (std + epsilon)

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
        normalizer = exped.sum(axis=axis) + epsilon
        if axis == 1:
            result = exped / normalizer.dimshuffle(0, 'x')
        if axis == 2:
            result = exped / normalizer.dimshuffle(0, 1, 'x')

        return result, T.zeros_like(var)

    return inner


def sigmoid(mean, var):
    mean_arg = mean / T.sqrt(1 + np.pi * var / 8)
    mean_ = T.nnet.sigmoid(mean_arg)

    a = 4 - 2 * np.sqrt(2)
    b = -np.log(np.sqrt(2) - 1)

    var_arg_1 = (
        a * (mean - b) /
        T.sqrt(1 + np.pi / (8 * a**2 * var + epsilon)))

    var_ = T.nnet.sigmoid(var_arg_1) - mean_ ** 2
    # It seems as if this aproximation yields non positive variances in corner
    # cases. We catch that here.
    var_ = T.maximum(1e-4, var_)

    return mean_, var_
