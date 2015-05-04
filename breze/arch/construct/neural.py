# -*- coding: utf-8 -*-


import theano.tensor as T

from breze.arch.construct.base import Layer

from breze.arch.construct import simple
from breze.arch.construct import sequential
from breze.arch.construct.layer.varprop import (
    simple as vp_simple, sequential as vp_sequential)


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

        self.outputs = inpt_mean, inpt_var


class Rnn(Layer):

    def __init__(self, inpt,
                 n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer='identity',
                 pooling=None,
                 declare=None, name=None):
        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.pooling = pooling

        super(Rnn, self).__init__(declare, name)

    def _forward(self):
        n_incoming = [self.n_inpt] + self.n_hiddens[:-1]
        n_outgoing = self.n_hiddens
        transfers = self.hidden_transfers

        n_time_steps, _, _ = self.inpt.shape

        self.layers = []
        x = self.inpt
        for n, m, t in zip(n_incoming, n_outgoing, transfers):
            x_flat = x.reshape((-1, n))
            pre_recurrent_flat = simple.AffineNonlinear(
                x_flat, n, m, t, declare=self.declare).output
            pre_recurrent = pre_recurrent_flat.reshape(
                (n_time_steps, -1, m))
            x = sequential.Recurrent(
                pre_recurrent, m, t, declare=self.declare).output

        x_flat = x.reshape((-1, m))
        output_flat = simple.AffineNonlinear(
            x_flat, m, self.n_output, self.out_transfer, declare=self.declare
            ).output
        output = output_flat.reshape((n_time_steps, -1, self.n_output))

        if self.pooling:
            self.pre_pooling = output
            self.output = sequential.Pooling(output, self.pooling).output
        else:
            self.output = output


class FastDropoutRnn(Layer):

    def __init__(self, inpt,
                 n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer='identity',
                 p_dropout_inpt=.2,
                 p_dropout_hiddens=.5,
                 p_dropout_hidden_to_out=None,
                 pooling=None,
                 declare=None, name=None):
        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.pooling = pooling

        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        self.p_dropout_hidden_to_out = p_dropout_hidden_to_out

        super(FastDropoutRnn, self).__init__(declare, name)

    def _forward(self):
        n_incoming = [self.n_inpt] + self.n_hiddens[:-1]
        n_outgoing = self.n_hiddens
        transfers = self.hidden_transfers
        p_dropouts = self.p_dropout_hiddens

        n_time_steps, _, _ = self.inpt.shape

        self.layers = []
        inpt_var = T.zeros_like(self.inpt)

        x_mean, x_var = vp_simple.FastDropout(
            self.inpt, inpt_var, self.p_dropout_inpt).outputs

        for m, n, t, d in zip(n_incoming, n_outgoing, transfers, p_dropouts):
            x_mean_flat = x_mean.reshape((-1, m))
            x_var_flat = x_var.reshape((-1, m))

            pre_rec_mean_flat, pre_rec_var_flat = vp_simple.AffineNonlinear(
                x_mean_flat, x_var_flat, m, n, t, declare=self.declare).outputs

            pre_rec_mean = pre_rec_mean_flat.reshape((n_time_steps, -1, n))
            pre_rec_var = pre_rec_var_flat.reshape((n_time_steps, -1, n))

            x_mean, x_var = vp_sequential.FDRecurrent(
                pre_rec_mean, pre_rec_var, n, t, p_dropout=d,
                declare=self.declare).outputs

        x_mean_flat = x_mean.reshape((-1, n))
        x_var_flat = x_var.reshape((-1, n))
        output_mean_flat, output_var_flat = vp_simple.AffineNonlinear(
            x_mean_flat, x_var_flat, n, self.n_output, self.out_transfer,
            declare=self.declare).outputs

        output_mean = output_mean_flat.reshape(
            (n_time_steps, -1, self.n_output))
        output_var = output_var_flat.reshape(
            (n_time_steps, -1, self.n_output))

        if self.pooling:
            raise NotImplemented()

        self.outputs = output_mean, output_var
