# -*- coding: utf-8 -*-

import theano.tensor as T

from breze.arch.construct.base import Layer

from breze.arch.util import get_named_variables


class Gauss(Layer):

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

    def forward(self, mean, var):
        var = T.sqrt(var ** 2 + 1e-8)
        self.exprs = get_named_variables(locals())
        self.output = [mean, var]
