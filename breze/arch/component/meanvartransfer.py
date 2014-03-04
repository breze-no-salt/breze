# -*- coding: utf-8 -*-


import numpy as np

import theano.tensor as T
import theano.tensor.extra_ops
from theano.tensor.shared_randomstreams import RandomStreams


# TODO I believe this module is obsolete, since there is
# breze.arch.component.varprop.transfer.


PI = np.array(np.pi, dtype=theano.config.floatX)
SQRT_2 = np.array(np.sqrt(2.), dtype=theano.config.floatX)
epsilon = np.array(1e-4, dtype=theano.config.floatX)


def normal_pdf(x, location=0, scale=1):
    location = T.cast(location, theano.config.floatX)
    SQRT_2_PI = np.sqrt(2 * PI)
    SQRT_2_PI = T.cast(SQRT_2_PI, theano.config.floatX)

    divisor = 2 * scale ** 2 + epsilon,
    divisor = T.cast(divisor, theano.config.floatX)

    exp_arg = -((x - location) ** 2) / divisor
    z = 1. / (SQRT_2_PI * scale + epsilon)

    return T.exp(exp_arg) * z


def normal_cdf(x, location=0, scale=1):
    location = T.cast(location, theano.config.floatX)
    scale = T.cast(scale, theano.config.floatX)

    div = T.sqrt(2 * scale ** 2 + epsilon)
    div = T.cast(div, theano.config.floatX)

    erf_arg = (x - location) / div
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
    mean_arg = mean / T.sqrt(1 + PI * var / 8)
    mean_ = T.nnet.sigmoid(mean_arg)

    a = 4 - 2 * SQRT_2
    b = -np.log(SQRT_2 - 1)

    # If we do not do the following (curiously) a will be float64 in all cases.
    a = T.cast(a, theano.config.floatX)
    b = T.cast(b, theano.config.floatX)

    var_arg_1 = (
        a *
        (mean - b) /
        T.sqrt(1 + PI / (8 * a**2 * var + epsilon)))

    var_ = T.nnet.sigmoid(var_arg_1) - mean_ ** 2
    # It seems as if this aproximation yields non positive variances in corner
    # cases. We catch that here.
    var_ = T.maximum(epsilon, var_)

    return mean_, var_
