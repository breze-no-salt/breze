# -*- coding: utf-8 -*-

import theano.tensor as T

from breze.arch.construct.base import Layer

from breze.arch.util import get_named_variables


class Gauss(Layer):
    """Gauss class.

    Layer that takes two inputs, representing the mean and the variance of a
    Gaussian and produces a single one, a sample from the corresponding
    distribution.
    """

    def __init__(self, rng=None, name=None):
        if rng is None:
            self.rng = T.shared_randomstreams.RandomStreams()
        else:
            self.rng = rng
        super(Gauss, self).__init__(name)

    def forward(self, mean, var):
        std = T.sqrt(var)
        noise = self.rng.normal(size=mean.shape)
        sample = mean + noise * std

        self.exprs = get_named_variables(locals())
        self.output = [sample]


class GaussProjection(Layer):
    """GaussProjection class.

    Layer that takes two inputs, mean and variance of a Gaussian. The variance
    can be negative and it is assured that it is positive afterwards by taking
    the square and the root after adding a small offset.
    """

    def forward(self, mean, var):
        var = T.sqrt(var ** 2 + 1e-8)
        self.exprs = get_named_variables(locals())
        self.output = [mean, var]
