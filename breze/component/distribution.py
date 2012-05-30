# -*- coding: utf-8 -*-

import collections
import types

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from math import pi

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm


class BernoulliDistribution(object):

    def __init__(self, seed=1010):
        self.srng = RandomStreams(seed=seed)

    @property
    def n_statistics(self):
        return 1

    def fixed_biases(self):
        # fixed_bias[statistic]
        return [False]

    def fixed_biases_values(self):
        # fixed_biases_values[statistic]
        return [0]

    def f(self, x):
        # x[node, sample] -> f[node, sample, statistic]
        fv = T.zeros((x.shape[0], x.shape[0], 1))
        fv[:, :, 0] = x
        return fv

    def lp(self, fac):
        # fac[node, sample, statistic] -> lpv[node, sample]
        return T.log(1 + fac[:, :, 0])

    def dlp(self, fac):
        # fac[node, sample, statistic] -> dlp[node, sample, statistic]
        return T.nnet.sigmoid(fac[:, :, :])

    def sampler(self, fac):
        # fac[node, sample, statistic] -> sample[node, sample]
        p = transfer.sigmoid(fac[:, :, 0])
        return self.srng.binomial(size=p.shape, n=1, p=p, 
                                  dtype=theano.config.floatX)

class NormalDistribution(object):

    def __init__(self, seed=1010):
        self.srng = RandomStreams(seed=seed)

    @property
    def n_statistics(self):
        return 2

    @property
    def fixed_bias(self):
        # fixed_bias[statistic]
        return [False, True]

    @property
    def fixed_bias_value(self):
        # fixed_bias_value[statistic]
        return [0, -1./2.]

    def f(self, x):
        # x[node, sample] -> f[node, sample, statistic]
        fv = T.zeros((x.shape[0], x.shape[0], 2))
        fv[:, :, 0] = x
        fv[:, :, 1] = T.sqr(x)
        return fv

    def lp(self, fac):
        # fac[node, sample, statistic] -> lpv[node, sample]
        return 1./2. * T.log(2. * pi) + fac[:, :, 0]

    def dlp(self, fac):
        # fac[node, sample, statistic] -> dlp[node, sample, statistic]
        dlpv = fac.clone()
        dlpv[:, :, 1] = 0
        return dlpv

    def sampler(self, fac):
        # fac[node, sample, statistic] -> sample[node, sample]
        return self.srng.normal(size=(fac.shape[0], fac.shape[1]), 
                                avg=fac[:, :, 0], std=1.0,  
                                dtype=theano.config.floatX)


