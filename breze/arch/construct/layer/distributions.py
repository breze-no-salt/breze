# -*- coding: utf-8 -*-

import theano.tensor as T
import numpy as np

from breze.arch.util import wild_reshape


def assert_no_time(X):
    if X.ndim == 2:
        return X
    if X.ndim != 3:
        raise ValueError('ndim must be 2 or 3, but it is %i' % X.ndim)
    return wild_reshape(X, (-1, X.shape[2]))


def recover_time(X, time_steps):
    return wild_reshape(X, (time_steps, -1, X.shape[1]))


def normal_logpdf(xs, means, vrs):
    energy = -(xs - means) ** 2 / (2 * vrs)
    partition_func = -T.log(T.sqrt(2 * np.pi * vrs))
    return partition_func + energy


class Distribution(object):

    def __init__(self, rng=None):
        if rng is None:
            self.rng = T.shared_randomstreams.RandomStreams()
        else:
            self.rng = rng

    def sample(self, epsilon=None):
        raise NotImplemented()

    def nll(self, X, inpt=None):
        raise NotImplemented()


class DiagGauss(Distribution):

    def __init__(self, mean, var, rng=None):
        self.mean = mean

        # This allows to use var with shape (1, 1, n)
        self.var = T.fill(mean, var)

        self.stt = T.concatenate((mean, self.var), -1)
        self.maximum = self.mean
        super(DiagGauss, self).__init__(rng)

    def sample(self, epsilon=None):
        mean_flat = assert_no_time(self.mean)
        var_flat = assert_no_time(self.var)

        if epsilon is None:
            noise = self.rng.normal(size=mean_flat.shape)
        else:
            noise = epsilon

        sample = mean_flat + T.sqrt(var_flat) * noise
        if self.mean.ndim == 3:
            return recover_time(sample, self.mean.shape[0])
        else:
            return sample

    def nll(self, X, inpt=None):
        var_offset = 1e-4
        var = self.var
        var += var_offset
        residuals = X - self.mean
        weighted_squares = -(residuals ** 2) / (2 * var)
        normalization = T.log(T.sqrt(2 * np.pi * var))
        ll = weighted_squares - normalization
        return -ll


class NormalGauss(Distribution):

    def __init__(self, shape, rng=None):
        self.shape = shape
        self.mean = T.zeros(shape)
        self.var = T.ones(shape)
        self.stt = T.concatenate((self.mean, self.var), -1)
        self.maximum = self.mean
        super(NormalGauss, self).__init__(rng)

    def sample(self):
        return self.rng.normal(size=self.shape)

    def nll(self, X, inpt=None):
        X_flat = X.flatten()
        nll = -normal_logpdf(X_flat, self.mean.flatten(), self.var.flatten())
        return nll.reshape(X.shape)


class Bernoulli(Distribution):

    def __init__(self, rate, rng=None):
        self.rate = rate
        self.stt = rate
        self.maximum = self.rate > 0.5
        super(Bernoulli, self).__init__(rng)

    def sample(self, epsilon=None):
        if epsilon is None:
            noise = self.rng.uniform(size=self.rate.shape)
        else:
            noise = epsilon
        sample = noise < self.rate
        return sample

    def nll(self, X, inpt=None):
        rate = self.rate
        rate *= 0.999
        rate += 0.0005
        return -(X * T.log(rate) + (1 - X) * T.log(1 - rate))
