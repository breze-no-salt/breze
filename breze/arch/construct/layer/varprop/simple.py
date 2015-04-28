# -*- coding: utf-8 -*-

import theano.tensor as T

from breze.arch.construct.base import Layer
from breze.arch.util import lookup, get_named_variables
from breze.arch.component.varprop import transfer

from breze.arch.construct.layer.simple import AffineNonlinear as AffineNonlinear_


def make_std(std):
    return (std ** 2 + 1e-8) ** 0.5


class AffineNonlinear(AffineNonlinear_):

    def forward(self, inpt_mean, inpt_var):
        Layer.forward(self, inpt_mean, inpt_var)
        P = self.parameters
        w, b = P.weights, P.bias
        weights = self.parameterized('weights', (self.n_inpt, self.n_output))
        bias = self.parameterized('bias', (self.n_output,)) if self.bias else 0

        pres_mean = T.dot(inpt_mean, weights) + bias
        pres_var = T.dot(inpt_var, weights ** 2)

        f_transfer = lookup(self.transfer, transfer)
        post_mean, post_var = f_transfer(pres_mean, pres_var)

        E = self.exprs = get_named_variables(locals())
        self.output = [post_mean, post_var]


class StochasticAffineNonlinear(AffineNonlinear_):

    def forward(self, inpt_mean, inpt_var):
        Layer.forward(self, inpt_mean, inpt_var)
        P = self.parameters
        weights_mean = self.parameterized(
            'weights_mean', (self.n_inpt, self.n_output))
        weights_std = self.parameterized(
            'weights_std', (self.n_inpt, self.n_output))
        if self.bias:
            bias_mean = self.parameterized('bias_mean', (self.n_output,))
            bias_std = self.parameterized('bias_std', (self.n_output,))
        else:
            bias_mean = bias_std = 0

        pres_mean = T.dot(inpt_mean, weights_mean) + bias_mean
        pres_var = (T.dot(inpt_mean ** 2, weights_std ** 2)
                    + T.dot(inpt_var, weights_mean ** 2)
                    + T.dot(inpt_var, weights_std ** 2)
                    + bias_std ** 2)

        f_transfer = lookup(self.transfer, transfer)
        post_mean, post_var = f_transfer(pres_mean, pres_var)

        E = self.exprs = get_named_variables(locals())
        self.output = [post_mean, post_var]


class AugmentVariance(Layer):

    def __init__(self, vari=1e-16, name=None):
        self.vari = vari
        super(AugmentVariance, self).__init__(name)

    def forward(self, inpt):
        super(AugmentVariance, self).forward(inpt)
        vari = T.zeros_like(inpt) + self.vari
        E = self.exprs = get_named_variables(locals())
        self.output = [inpt, vari]


class DiscardVariance(Layer):

    def forward(self, mean, vari):
        super(DiscardVariance, self).forward(mean, vari)
        self.exprs = {'mean': mean}
        self.output = mean,


class FastDropout(Layer):

    def __init__(self, p_dropout, name=None):
        if not (0 < p_dropout <= 1):
            raise ValueError('p_dropout has to lie in (0, 1]')

        self.p_dropout = p_dropout
        super(FastDropout, self).__init__(name)

    def forward(self, inpt_mean, inpt_var):
        p_keep = 1 - self.p_dropout
        output_mean = inpt_mean * p_keep
        dropout_var = p_keep * (1 - p_keep)
        output_var = (inpt_mean ** 2 * dropout_var
                      + p_keep ** 2 * inpt_var
                      + dropout_var * inpt_var)

        E = self.exprs = get_named_variables(locals())
        self.output = [output_mean, output_var]
