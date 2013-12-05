# -*- coding: utf-8 -*-

"""Module that contains transfer functions for variance propagation, working on
Theano variables.

Each transfer function has the signature::

    m2, s2 = f(m1, s1)

where ``f`` is the transfer function, ``m1`` and ``s2`` are the pre-synaptic
mean and variance respectively; ``m2`` and ``s2`` are the post-synaptic means.
"""

import numpy as np

import theano.tensor as T
import theano.tensor.extra_ops
from theano.tensor.nnet import softmax as _softmax, sigmoid as _sigmoid
from theano.tensor.shared_randomstreams import RandomStreams

from breze.arch.component.distributions import normal


PI = np.array(np.pi, dtype=theano.config.floatX)
SQRT_2 = np.array(np.sqrt(2.), dtype=theano.config.floatX)
epsilon = np.array(1e-4, dtype=theano.config.floatX)


def safe_sigmoid(x):
    """Return the sigmoid with result truly between 0 and 1."""
    y = _sigmoid(x)
    return T.clip(y, 1e-7, 1 - 1e-7)


def safe_softmax(x):
    """Return the softmax with result truly between 0 and 1."""
    y = _softmax(x)
    y = T.clip(y, 1e-7, 1 - 1e-7)
    return y


def identity(mean, var):
    """Return the mean and variance unchanged.

    Parameters
    ----------

    mean : Theano variable
        Theano variable of the shape ``s``.

    var : Theano variable
        Theano variable of the shape ``s``.

    Returns
    -------

    mean_ : Theano variable
        Theano variable of the shape ``r``.

    var_ : Theano variable
        Theano variable of the shape ``r``.
    """
    return mean, var


def rectifier(mean, var):
    """Return the mean and variance of a Gaussian distributed random variable,
    described by its mean and variacne, after passing it through a rectified
    linear unit.

    Parameters
    ----------

    mean : Theano variable
        Theano variable of the shape ``s``.

    var : Theano variable
        Theano variable of the shape ``s``.

    Returns
    -------

    mean_ : Theano variable
        Theano variable of the shape ``r``.

    var_ : Theano variable
        Theano variable of the shape ``r``.
    """
    std = T.sqrt(var)
    ratio = mean / (std + epsilon)

    mean_ = normal.cdf(ratio) * mean + normal.pdf(ratio) * std

    A = mean * std * normal.pdf(ratio)
    B = (mean ** 2 + std ** 2) * normal.cdf(ratio)
    exp_of_squared = A + B

    mean_ = T.clip(mean_, 1e-8, 100)

    var_ = exp_of_squared - mean_ ** 2 + 0.05
    var_ = T.clip(var_, 1e-8, 100)


    return mean_, var_


def make_sampling_transfer(f, axis=1, rng=None):
    if rng is None:
        rng = RandomStreams()

    def inner(mean, var):
        # Generate samples of the distribution.
        samples = rng.normal(size=mean.shape)
        std = T.sqrt(var)
        samples = samples * std + mean

        if axis == 1:
            result = f(samples)  # XXX
        elif axis == 2:
            samples_flat = samples.reshape((samples.shape[0] * samples.shape[1], samples.shape[2]))
            result_flat = f(samples_flat)
            result = result_flat.reshape(samples.shape)

        return result, T.zeros_like(var) + 0.1

    return inner


sampling_softmax = make_sampling_transfer(safe_softmax)
sampling_sigmoid = make_sampling_transfer(safe_sigmoid)


def sigmoid(mean, var):
    """Return the mean and variance of a Gaussian distributed random variable,
    described by its mean and variacne, after passing it through a logistic
    sigmoid.

    Parameters
    ----------

    mean : Theano variable
        Theano variable of the shape ``s``.

    var : Theano variable
        Theano variable of the shape ``s``.

    Returns
    -------

    mean_ : Theano variable
        Theano variable of the shape ``r``.

    var_ : Theano variable
        Theano variable of the shape ``r``.
    """
    make_mean = lambda m, s2: T.nnet.sigmoid(m / (T.sqrt(1 + PI * s2 / 8)))

    a = 4 - 2 * SQRT_2
    b = -np.log(SQRT_2 - 1)

    # If we do not do the following (curiously) a will be float64 in all cases.
    a = T.cast(a, theano.config.floatX)
    b = T.cast(b, theano.config.floatX)

    mean_ = make_mean(mean, var)
    var_ = make_mean(a * (mean - b), a ** 2 * var) - mean_ ** 2

    #var_arg_1 = (
    #    a *
    #    (mean - b) / T.sqrt(1 + PI / (8 * a ** 2 * var + epsilon)))

    #var_ = T.nnet.sigmoid(var_arg_1) - mean_ ** 2
    # It seems as if this aproximation yields non positive variances in corner
    # cases. We catch that here.
    #var_ = T.maximum(epsilon, var_)

    # This approximation might yield 0 or 1. This is however bad for subsequent
    # losses such as the negative cross entropy; also, the sigmoid function will
    # *nerver* do this. Thus we clip it. The value of 1e-6 has been chosen to
    # work well on float32.
    mean_ = T.clip(mean_, 1e-7, 1 - 1e-7)

    return mean_, var_


def tanh(mean, var):
    """Return the mean and variance of a Gaussian distributed random variable,
    described by its mean and variacne, after passing it through a tangent
    hyperbolicus.

    Note
    ----

    Implementation is done by a rescaling, shifting and appllying ``sigmoid``.


    Parameters
    ----------

    mean : Theano variable
        Theano variable of the shape ``s``.

    var : Theano variable
        Theano variable of the shape ``s``.

    Returns
    -------

    mean_ : Theano variable
        Theano variable of the shape ``r``.

    var_ : Theano variable
        Theano variable of the shape ``r``.
    """
    mean_, var_ = sigmoid(mean, var)
    return mean_ * 2 - 1, 4 * var_
