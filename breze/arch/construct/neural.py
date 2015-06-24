# -*- coding: utf-8 -*-


from collections import namedtuple

import theano.tensor as T

from breze.arch.component import transfer as _transfer
from breze.arch.construct.base import Layer
from breze.arch.construct import simple
from breze.arch.construct import sequential
from breze.arch.construct.layer.varprop import (
    simple as vp_simple, sequential as vp_sequential)
from breze.arch.util import lookup


def wild_reshape(tensor, shape):
    n_m1 = shape.count(-1)
    if n_m1 > 1:
        raise ValueError(' only one -1 allowed in shape')
    elif n_m1 == 1:
        rest = tensor.size
        for s in shape:
            if s != -1:
                rest = rest // s
        shape = tuple(i if i != -1 else rest for i in shape)
    return tensor.reshape(shape)


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

        n_inpts = [self.n_inpt] + self.n_hiddens
        n_outputs = self.n_hiddens + [self.n_output]
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
                 dropout_parameterized=False,
                 declare=None, name=None):
        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        self.dropout_parameterized = dropout_parameterized

        super(FastDropoutMlp, self).__init__(declare, name)

    def _forward(self):
        self.fd_layers = []
        self.layers = []

        n_inpts = [self.n_inpt] + self.n_hiddens
        n_outputs = self.n_hiddens + [self.n_output]
        transfers = self.hidden_transfers + [self.out_transfer]
        p_dropouts = [self.p_dropout_inpt] + self.p_dropout_hiddens

        inpt_mean = self.inpt
        inpt_var = T.zeros_like(inpt_mean) + 1e-16

        for n, m, t, p in zip(n_inpts, n_outputs, transfers, p_dropouts):
            if self.dropout_parameterized:
                p = self.declare((1,))

                p = T.nnet.sigmoid(p) * 0.49 + 0.01

            fd_layer = vp_simple.FastDropout(
                inpt_mean, inpt_var, p, declare=self.declare)
            mean, vari = fd_layer.outputs
            self.fd_layers.append(fd_layer)

            layer = vp_simple.AffineNonlinear(
                mean, vari, n, m, t, declare=self.declare)

            self.layers.append(layer)

            inpt_mean, inpt_var = layer.outputs

        self.output = T.concatenate((inpt_mean, inpt_var), 1)
        self.outputs = inpt_mean, inpt_var


class Rnn(Layer):

    HiddenLayer = namedtuple('HiddenLayer', 'affine recurrent'.split())
    OutputLayer = namedtuple('OutputLayer', ['affine'])

    @property
    def hidden_layers(self):
        return [i for i in self.layers if isinstance(i, self.HiddenLayer)]

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
        transfers = self.hidden_transfers
        transfers =[lookup(i, _transfer) for i in transfers]
        transfer_insizes = [getattr(i, 'in_size', 1) for i in transfers]
        transfer_outsizes = [1] + [getattr(i, 'out_size', 1) for i in transfers]

        n_incoming = [self.n_inpt] + self.n_hiddens[:-1]
        n_outgoing = self.n_hiddens

        n_time_steps, _, _ = self.inpt.shape

        self.layers = []
        x = self.inpt
        for n, m, t, tis, tos in zip(n_incoming, n_outgoing, transfers,
                                     transfer_insizes, transfer_outsizes):
            x_flat = x.reshape((-1, n))

            affine = simple.AffineNonlinear(
                x_flat, n * tos, m * tis, lambda x: x, declare=self.declare)
            pre_recurrent_flat = affine.output

            pre_recurrent = pre_recurrent_flat.reshape(
                (n_time_steps, -1, m * tis))

            recurrent = sequential.Recurrent(
                pre_recurrent, m * t.out_size, t, declare=self.declare)
            x = recurrent.output

            self.layers.append(self.HiddenLayer(affine, recurrent))

        x_flat = x.reshape((-1, m * t.out_size))
        out_transfer = lookup(self.out_transfer, _transfer)
        out_in_size = getattr(out_transfer, 'in_size', 1)
        output_affine = simple.AffineNonlinear(
            x_flat, m, self.n_output * out_in_size , out_transfer,
            declare=self.declare
            )

        self.layers.append(self.OutputLayer(affine))

        output = output_affine.output.reshape((n_time_steps, -1, self.n_output))

        if self.pooling:
            self.pre_pooling = output
            self.output = sequential.Pooling(output, self.pooling).output
        else:
            self.output = output


class FastDropoutRnn(Layer):

    InputLayer = namedtuple('InputLayer', ['fast_dropout'])
    HiddenLayer = namedtuple('HiddenLayer',
                             'affine recurrent'.split())
    OutputLayer = namedtuple('OutputLayer', 'fast_dropout affine'.split())

    @property
    def hidden_layers(self):
        return [i for i in self.layers if isinstance(i, self.HiddenLayer)]

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

    def _make_rec_layer(self, x_mean, x_var, n_inpt, n_output, transfer,
                        p_dropout):
        n_time_steps, _, _ = self.inpt.shape
        x_mean_flat = wild_reshape(x_mean, (-1, n_inpt))
        x_var_flat = wild_reshape(x_var, (-1, n_inpt))

        affine = vp_simple.AffineNonlinear(
            x_mean_flat, x_var_flat, n_inpt, n_output, 'identity',
            declare=self.declare)
        pre_rec_mean_flat, pre_rec_var_flat = affine.outputs

        pre_rec_mean = wild_reshape(pre_rec_mean_flat,
                                    (n_time_steps, -1, n_output))
        pre_rec_var = wild_reshape(pre_rec_var_flat,
                                   (n_time_steps, -1, n_output))

        if p_dropout == 'parameterized':
            p_dropout = self.declare((1,))
            p_dropout = T.nnet.sigmoid(p_dropout) * 0.49 + 0.01

        recurrent = vp_sequential.FDRecurrent(
            pre_rec_mean, pre_rec_var, n_output, transfer, p_dropout=p_dropout,
            declare=self.declare)
        x_mean, x_var = recurrent.outputs
        layer = self.HiddenLayer(affine, recurrent)
        return x_mean, x_var, layer

    def _forward(self):
        n_incoming = [self.n_inpt] + self.n_hiddens[:-1]
        n_outgoing = self.n_hiddens
        transfers = self.hidden_transfers
        p_dropouts = self.p_dropout_hiddens

        n_time_steps, _, _ = self.inpt.shape

        self.layers = []
        inpt_var = T.zeros_like(self.inpt)

        if self.p_dropout_inpt == 'parameterized':
            p_dropout_inpt = self.declare((1,))
            p_dropout_inpt = T.nnet.sigmoid(p_dropout_inpt) * 0.49 + 0.01
        else:
            p_dropout_inpt = self.p_dropout_inpt

        fd_layer = vp_simple.FastDropout(
            self.inpt, inpt_var, p_dropout_inpt)
        self.layers.append(self.InputLayer(fd_layer))
        x_mean, x_var = fd_layer.outputs

        for m, n, t, d in zip(n_incoming, n_outgoing, transfers, p_dropouts):
            x_mean, x_var, layer = self._make_rec_layer(
                x_mean, x_var, m, n, t, d)
            self.layers.append(layer)

        x_mean_flat = wild_reshape(x_mean, (-1, n))
        x_var_flat = wild_reshape(x_var, (-1, n))
        if self.p_dropout_hidden_to_out == 'parameterized':
            p_dropout_hidden_to_out = self.declare((1,))
            p_dropout_hidden_to_out = T.nnet.sigmoid(
                p_dropout_hidden_to_out) * 0.49 + 0.01
        else:
            p_dropout_hidden_to_out = self.p_dropout_hidden_to_out

        fd = vp_simple.FastDropout(
            x_mean_flat, x_var_flat, p_dropout_hidden_to_out)
        x_mean_flat, x_var_flat = fd.outputs
        affine = vp_simple.AffineNonlinear(
            x_mean_flat, x_var_flat, n, self.n_output, self.out_transfer,
            declare=self.declare)
        output_mean_flat, output_var_flat = affine.outputs
        self.layers.append(self.OutputLayer(fd, affine))

        output_mean = wild_reshape(
            output_mean_flat, (n_time_steps, -1, self.n_output))
        output_var = wild_reshape(
            output_var_flat, (n_time_steps, -1, self.n_output))

        if self.pooling:
            raise NotImplemented()

        self.output = T.concatenate([output_mean, output_var], 2)
        self.outputs = output_mean, output_var


class BidirectFastDropoutRnn(FastDropoutRnn):

    HiddenLayer = namedtuple(
        'HiddenLayer',
        'affine recurrent_forward recurrent_backward'.split())

    def _make_rec_layer(self, x_mean, x_var, n_inpt, n_output, transfer,
                        p_dropout):
        n_time_steps, _, _ = self.inpt.shape
        x_mean_flat = wild_reshape(x_mean, (-1, n_inpt))
        x_var_flat = wild_reshape(x_var, (-1, n_inpt))

        affine = vp_simple.AffineNonlinear(
            x_mean_flat, x_var_flat, n_inpt, n_output, 'identity',
            declare=self.declare)
        pre_rec_mean_flat, pre_rec_var_flat = affine.outputs

        pre_rec_mean = wild_reshape(pre_rec_mean_flat,
                                    (n_time_steps, -1, n_output))
        pre_rec_var = wild_reshape(pre_rec_var_flat,
                                   (n_time_steps, -1, n_output))

        if p_dropout == 'parameterized':
            p_dropout = self.declare((1,))
            p_dropout = T.nnet.sigmoid(p_dropout) * 0.49 + 0.01

        recurrent_fw = vp_sequential.FDRecurrent(
            pre_rec_mean, pre_rec_var, n_output, transfer, p_dropout=p_dropout,
            declare=self.declare)
        recurrent_bw = vp_sequential.FDRecurrent(
            pre_rec_mean[::-1], pre_rec_var[::-1], n_output, transfer,
            p_dropout=p_dropout,
            declare=self.declare)

        x_mean = recurrent_fw.outputs[0] + recurrent_bw.outputs[0]
        x_var = recurrent_fw.outputs[1] + recurrent_bw.outputs[1]

        layer = self.HiddenLayer(affine, recurrent_fw, recurrent_bw)
        return x_mean, x_var, layer
