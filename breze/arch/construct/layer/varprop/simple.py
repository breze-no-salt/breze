# -*- coding: utf-8 -*-

import theano.tensor as T

from breze.arch.construct.base import Layer
from breze.arch.util import lookup, get_named_variables
from breze.arch.component.varprop import transfer


def make_std(std):
    return (std ** 2 + 1e-8) ** 0.5


class AffineNonlinear(Layer):

    def __init__(self, inpt_mean, inpt_var, n_inpt, n_output,
                 transfer='identity', use_bias=True, declare=None, name=None):
        self.inpt_mean = inpt_mean
        self.inpt_var = inpt_var
        self.n_inpt = n_inpt
        self.n_output = n_output
        self.transfer = transfer
        self.use_bias = use_bias
        super(AffineNonlinear, self).__init__(declare, name)

    def _forward(self):
        w = self.weights = self.declare((self.n_inpt, self.n_output))
        b = self.bias = self.declare(self.n_output)

        self.pres_mean = T.dot(self.inpt_mean, w)
        if self.use_bias:
            self.pres_mean += b

        self.pres_var = T.dot(self.inpt_var, w ** 2)

        f_transfer = lookup(self.transfer, transfer)
        self.post_mean, self.post_var = f_transfer(
            self.pres_mean, self.pres_var)

        self.outputs = self.post_mean, self.post_var


#class StochasticAffineNonlinear(AffineNonlinear_):
#
#    def forward(self, inpt_mean, inpt_var):
#        Layer.forward(self, inpt_mean, inpt_var)
#        P = self.parameters
#        weights_mean = self.parameterized(
#            'weights_mean', (self.n_inpt, self.n_output))
#        weights_std = self.parameterized(
#            'weights_std', (self.n_inpt, self.n_output))
#        if self.bias:
#            bias_mean = self.parameterized('bias_mean', (self.n_output,))
#            bias_std = self.parameterized('bias_std', (self.n_output,))
#        else:
#            bias_mean = bias_std = 0
#
#        pres_mean = T.dot(inpt_mean, weights_mean) + bias_mean
#        pres_var = (T.dot(inpt_mean ** 2, weights_std ** 2)
#                    + T.dot(inpt_var, weights_mean ** 2)
#                    + T.dot(inpt_var, weights_std ** 2)
#                    + bias_std ** 2)
#
#        f_transfer = lookup(self.transfer, transfer)
#        post_mean, post_var = f_transfer(pres_mean, pres_var)
#
#        E = self.exprs = get_named_variables(locals())
#        self.output = [post_mean, post_var]


class FastDropout(Layer):

    def __init__(self, inpt_mean, inpt_var, p_dropout, declare=None, name=None):
        if isinstance(p_dropout, float) and not (0 < p_dropout <= 1):
            raise ValueError('p_dropout has to lie in (0, 1]')

        self.inpt_mean = inpt_mean
        self.inpt_var = inpt_var
        self.p_dropout = p_dropout
        super(FastDropout, self).__init__(declare, name)

    def _forward(self):
        p_keep = 1 - self.p_dropout
        self.output_mean = self.inpt_mean * p_keep
        dropout_var = p_keep * (1 - p_keep)
        self.output_var = (self.inpt_mean ** 2 * dropout_var
                      + p_keep ** 2 * self.inpt_var
                      + dropout_var * self.inpt_var)

        self.outputs = self.output_mean, self.output_var
