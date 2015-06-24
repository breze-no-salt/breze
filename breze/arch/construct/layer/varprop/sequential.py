# -*- coding: utf-8 -*-

import theano.tensor as T

from breze.arch.component.varprop import transfer as _transfer
from breze.arch.construct.base import Layer
from breze.arch.model.varprop.rnn import (recurrent_layer,
                                          recurrent_layer_stateful)
from breze.arch.util import lookup


class FDRecurrent(Layer):

    def __init__(self, inpt_mean, inpt_var, n_inpt, transfer, p_dropout,
                 declare=None, name=None):
        self.inpt_mean = inpt_mean
        self.inpt_var = inpt_var
        self.n_inpt = n_inpt
        self.transfer = transfer
        self.p_dropout = p_dropout
        super(FDRecurrent, self).__init__(declare, name)

    def _forward(self):
        f = lookup(self.transfer, _transfer)
        n, m = self.n_inpt, self.n_inpt

        # If a transfer function has differing dimensionalities in its domain
        # and co-domain, it can be specified by its ``in_size`` and ``out_size``
        # attributes. Defaulting to not to.
        n *= getattr(f, 'in_size', 1)
        m *= getattr(f, 'out_size', 1)

        self.initial_mean = self.declare((m,))
        self.initial_std = self.declare((m,))
        self.weights = self.declare((m, n))

        if getattr(f, 'stateful', False):
            res = recurrent_layer_stateful(
                self.inpt_mean, self.inpt_var,
                self.weights,
                f,
                self.initial_mean, self.initial_std ** 2 + 1e-8,
                self.p_dropout)
            (self.state_mean, self.state_var,
             self.output_in_mean, self.output_in_var,
             self.output_mean, self.output_var) = res
        else:
            res = recurrent_layer(
                self.inpt_mean, self.inpt_var,
                self.weights,
                f,
                self.initial_mean, self.initial_std ** 2 + 1e-8,
                self.p_dropout)
            (self.output_in_mean, self.output_in_var,
             self.output_mean, self.output_var) = res

        self.outputs = self.output_mean, self.output_var


# TODO this is only here for the proof of concept, it should go to a better
# place and ideally be combined with recurrent_layer
import theano
from theano.tensor.extra_ops import repeat


def fawn_recurrent(
    inpt_mean, inpt_var, weights_mean, weights_var,
    f,
    initial_mean, initial_var):

    f_transfer = lookup(f, transfer_)
    def step(inpt_mean, inpt_var, him_m1, hiv_m1, hom_m1, hov_m1):
        wm, wv = weights_mean, weights_var

        pres_mean = T.dot(inpt_mean, wm)
        pres_var = (T.dot(inpt_mean ** 2, wv)
                    + T.dot(inpt_var, wm ** 2)
                    + T.dot(inpt_var, wv)
                    )

        post_mean, post_var = f_transfer(pres_mean, pres_var)
        return pres_mean, pres_var, post_mean, post_var


    if initial_mean.ndim == 1:
        initial_mean = repeat(
            initial_mean.dimshuffle('x', 0), inpt_mean.shape[1], axis=0)
    if initial_var.ndim == 1:
        initial_var = repeat(
            initial_var.dimshuffle('x', 0), inpt_mean.shape[1], axis=0)

    (hidden_in_mean_rec, hidden_in_var_rec, hidden_mean_rec, hidden_var_rec), _ = theano.scan(
        step,
        sequences=[inpt_mean, inpt_var],
        outputs_info=[T.zeros_like(inpt_mean[0]),
                      T.zeros_like(inpt_mean[0]),
                      initial_mean,
                      initial_var])

    return (hidden_in_mean_rec, hidden_in_var_rec,
            hidden_mean_rec, hidden_var_rec)
