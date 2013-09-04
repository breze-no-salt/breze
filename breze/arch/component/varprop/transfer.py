# -*- coding: utf-8 -*-

import numpy as np

import theano.tensor as T
import theano.tensor.extra_ops
from theano.tensor.nnet import softmax
from theano.tensor.shared_randomstreams import RandomStreams

from breze.arch.component.distributions import normal


PI = np.array(np.pi, dtype=theano.config.floatX)
SQRT_2 = np.array(np.sqrt(2.), dtype=theano.config.floatX)
epsilon = np.array(1e-4, dtype=theano.config.floatX)


def identity(mean, var):
    return mean, var


def rectifier(mean, var):
    std = T.sqrt(var)
    ratio = mean / (std + epsilon)

    mean_ = normal.cdf(ratio) * mean + normal.pdf(ratio) * std

    A = mean * std * normal.pdf(ratio)
    B = (mean ** 2 + std ** 2) * normal.cdf(ratio)
    exp_of_squared = A + B

    var_ = exp_of_squared - mean_ ** 2
    return mean_, var_


def make_sampling_softmax(axis=1, rng=None):
    if rng is None:
        rng = RandomStreams()

    def inner(mean, var):
        # Generate samples of the distribution.
        samples = rng.normal(size=mean.shape)
        std = T.sqrt(var)
        samples = samples * std + mean

        if axis == 1:
            result = softmax(samples) # XXX
            result.name = 'susp1'
        if axis == 2:
            samples_flat = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2]))
            result_flat = softmax(samples_flat)
            result = result.reshape(samples.shape)

        return result, T.zeros_like(var)

    return inner

sampling_softmax = make_sampling_softmax()


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
        (mean - b) / T.sqrt(1 + PI / (8 * a ** 2 * var + epsilon)))

    var_ = T.nnet.sigmoid(var_arg_1) - mean_ ** 2
    # It seems as if this aproximation yields non positive variances in corner
    # cases. We catch that here.
    var_ = T.maximum(epsilon, var_)

    return mean_, var_
