# -*- coding: utf-8 -*-

import collections
import types

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from math import pi

from . import transfer, distance, norm


class Distribution(object):
    """A distribution"""

    def __init__(self):
        raise NotImplementedError()

    @property
    def n_statistics(self):
        """Number of sufficient statistics for exponential family harmoniums"""
        raise NotImplementedError()

    @property
    def fixed_bias(self):
        """List of boolean values, one for each statistic determining if bias value
        is fixed for that statistic"""
        # fixed_bias[statistic]
        raise NotImplementedError()

    @property
    def fixed_bias_value(self):
        """List of fixed bias values, one for each statistic. Value is only
        considered for statistics where fixed_bias[statistic] is True."""
        # fixed_biases_values[statistic]
        raise NotImplementedError()

    def f(self, x):
        """Function returning sufficient statistics for given node values,
        i.e. x[node, sample] -> f[node, sample, statistic]"""
        raise NotImplementedError()

    def lp(self, fac):
        """Function returning log-partition function for given statistic
        factors fac, i.e. fac[node, sample, statistic] -> lp[node, sample]"""
        raise NotImplementedError()

    def dlp(self, fac):
        """Function returning partial derivative of log-partition function
        with regard to all statistics, i.e. 
        fac[node, sample, statistic] -> dlp[node, sample, statistic]
        where dlp[node, sample, statistic] is the derivative of 
        lp[node, sample] with regard to statistic."""
        raise NotImplementedError()

    def sampler(self, fac, final_gibbs_sample):
        """Function sampling from the distribution given statistic factors fac,
        i.e. fac[node, sample, statistic] -> sample[node, sample].
        
        :param final_gibbs_sample: Is True on the final iteration during
                                   Gibbs sampling.
        """
        raise NotImplementedError()


class BernoulliDistribution(Distribution):
    """Bernoulli distribution"""

    def __init__(self, seed=1010):
        self.srng = RandomStreams(seed=seed)

    @property
    def n_statistics(self):
        return 1

    @property
    def fixed_bias(self):
        # fixed_bias[statistic]
        return [False]

    @property
    def fixed_bias_value(self):
        # fixed_biases_values[statistic]
        return [0]

    def f(self, x):
        # x[node, sample] -> f[node, sample, statistic]
        fv = T.zeros((x.shape[0], x.shape[1], 1))
        fv = T.set_subtensor(fv[:, :, 0], x)
        return fv

    def lp(self, fac):
        # fac[node, sample, statistic] -> lpv[node, sample]
        return T.log(1 + fac[:, :, 0])

    def dlp(self, fac):
        # fac[node, sample, statistic] -> dlp[node, sample, statistic]
        return T.nnet.sigmoid(fac[:, :, :])

    def sampler(self, fac, final_gibbs_sample):
        # fac[node, sample, statistic] -> sample[node, sample]
        p = transfer.sigmoid(fac[:, :, 0])
        if final_gibbs_sample:
            return p
        else:
            return self.srng.binomial(size=p.shape, n=1, p=p, 
                                      dtype=theano.config.floatX)


class NormalDistribution(Distribution):

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
        fv = T.zeros((x.shape[0], x.shape[1], 2))
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

    def sampler(self, fac, final_gibbs_sample):
        # fac[node, sample, statistic] -> sample[node, sample]
        mean = fac[:, :, 0]
        if final_gibbs_sample:
            return mean
        else:
            return self.srng.normal(size=mean.shape, 
                                    avg=mean, std=1.0,  
                                    dtype=theano.config.floatX)
