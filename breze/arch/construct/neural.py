# -*- coding: utf-8 -*-


import theano.tensor as T

from breze.arch.construct.base import Layer

from breze.arch.construct import simple
from breze.arch.construct.layer.varprop import simple as vp_simple


class Mlp(Layer):

    def __init__(self, inpt, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, declare=None, name=None):
        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer

        super(Mlp, self).__init__(declare, name)

    def _forward(self):
        self.layers = []

        n_inpts = [self.n_inpt] + [self.n_hiddens[-1]]
        n_outputs = self.n_hiddens[1:] + [self.n_output]
        transfers = self.hidden_transfers + [self.out_transfer]

        inpt = self.inpt
        for n, m, t in zip(n_inpts, n_outputs, transfers):
            layer = simple.AffineNonlinear(inpt, n, m, t, declare=self.declare)
            self.layers.append(layer)
            inpt = layer.output

        self.output = inpt


class FastDropoutMlp(Layer):

    def __init__(self, inpt, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer,
                 p_dropout_inpt,
                 p_dropout_hiddens,
                 declare=None, name=None):
        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens

        super(FastDropoutMlp, self).__init__(declare, name)

    def _forward(self):
        self.fd_layers = []
        self.layers = []

        n_inpts = [self.n_inpt] + [self.n_hiddens[-1]]
        n_outputs = self.n_hiddens[1:] + [self.n_output]
        transfers = self.hidden_transfers + [self.out_transfer]
        p_dropouts = [self.p_dropout_inpt] + self.p_dropout_hiddens

        inpt_mean = self.inpt
        inpt_var = T.zeros_like(inpt_mean) + 1e-16

        for n, m, t, p in zip(n_inpts, n_outputs, transfers, p_dropouts):
            fd_layer = vp_simple.FastDropout(
                inpt_mean, inpt_var, p, declare=self.declare)
            mean, vari = fd_layer.outputs
            self.fd_layers.append(fd_layer)

            layer = vp_simple.AffineNonlinear(
                mean, vari, n, m, t, declare=self.declare)

            self.layers.append(layer)

            inpt_mean, inpt_var = layer.outputs

        self.output = T.concatenate((inpt_mean, inpt_var),1)
        self.outputs = inpt_mean, inpt_var
