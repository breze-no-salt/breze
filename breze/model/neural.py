# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from ..util import ParameterSet, Model, lookup
from ..component import transfer, distance


class TwoLayerPerceptron(Model):

    def __init__(self, n_inpt, n_hidden, n_output,
                 hidden_transfer, out_transfer, loss):
        self.n_inpt = n_inpt
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.hidden_transfer = hidden_transfer
        self.out_transfer = out_transfer
        self.loss = loss

        super(TwoLayerPerceptron, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(
            self.n_inpt, self.n_hidden, self.n_output)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        self.exprs = self.make_exprs(
            T.matrix('inpt'), T.matrix('target'),
            self.parameters.in_to_hidden, self.parameters.hidden_to_out,
            self.parameters.hidden_bias, self.parameters.out_bias,
            self.hidden_transfer, self.out_transfer, self.loss)

    @staticmethod
    def get_parameter_spec(n_inpt, n_hidden, n_output):
        return dict(in_to_hidden=(n_inpt, n_hidden),
                    hidden_to_out=(n_hidden, n_output),
                    hidden_bias=n_hidden,
                    out_bias=n_output)

    @staticmethod
    def make_exprs(inpt, target, in_to_hidden, hidden_to_out,
                   hidden_bias, out_bias,
                   hidden_transfer, output_transfer, loss):

        f_hidden = lookup(hidden_transfer, transfer)
        f_output = lookup(output_transfer, transfer)
        f_loss = lookup(loss, distance)

        hidden_in = T.dot(inpt, in_to_hidden) + hidden_bias
        hidden = f_hidden(hidden_in)

        output_in = T.dot(hidden, hidden_to_out) + out_bias
        output = f_output(output_in)

        loss = f_loss(target, output, axis=1).sum()

        return {
            'inpt': inpt,
            'target': target,
            'hidden_in': hidden_in,
            'hidden': hidden,
            'output_in': output_in,
            'output': output,
            'loss': loss
        }


class MultiLayerPerceptron(Model):

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss):
        if len(n_hiddens) != len(hidden_transfers):
            raise ValueError('n_hiddens and hidden_transfers have to be of the'
                             'same length')
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output

        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.loss = loss

        super(MultiLayerPerceptron, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(
            self.n_inpt, self.n_hiddens, self.n_output)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        hidden_to_hiddens = [getattr(self.parameters, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(self.parameters, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        self.exprs = self.make_exprs(
            T.matrix('inpt'), T.matrix('target'),
            self.parameters.in_to_hidden,
            hidden_to_hiddens,
            self.parameters.hidden_to_out,
            hidden_biases,
            self.parameters.out_bias,
            self.hidden_transfers, self.out_transfer, self.loss)

    @staticmethod
    def get_parameter_spec(n_inpt, n_hiddens, n_output):
        spec = dict(in_to_hidden=(n_inpt, n_hiddens[0]),
                    hidden_to_out=(n_hiddens[-1], n_output),
                    hidden_bias_0=n_hiddens[0],
                    out_bias=n_output)

        zipped = zip(n_hiddens[:-1], n_hiddens[1:])
        spec['hidden_bias_0'] = n_hiddens[0]
        for i, (inlayer, outlayer) in enumerate(zipped):
            spec['hidden_to_hidden_%i' % i] = (inlayer, outlayer)
            spec['hidden_bias_%i' % (i + 1)] = outlayer

        return spec

    @staticmethod
    def make_exprs(inpt, target, in_to_hidden,
                   hidden_to_hiddens,
                   hidden_to_out,
                   hidden_biases,
                   out_bias,
                   hidden_transfers, output_transfer, loss):
        exprs = {}

        f_hidden = lookup(hidden_transfers[0], transfer)
        hidden_in = exprs['hidden_in_0'] = T.dot(inpt, in_to_hidden) + hidden_biases[0]
        hidden = exprs['hidden_0'] =  f_hidden(hidden_in)

        zipped = zip(hidden_to_hiddens, hidden_biases[1:], hidden_transfers[1:])
        for i, (w, b, t) in enumerate(zipped):
            hidden_m1 = hidden
            hidden_in = T.dot(hidden_m1, w) + b
            f = lookup(t, transfer)
            hidden = f(hidden_in)
            exprs['hidden_in_%i' % (i + 1)] = hidden_in
            exprs['hidden_%i' % (i + 1)] = hidden

        f_output = lookup(output_transfer, transfer)
        output_in = T.dot(hidden, hidden_to_out) + out_bias
        output = f_output(output_in)

        f_loss = lookup(loss, distance)

        loss = f_loss(target, output, axis=1).sum()

        exprs.update({
            'inpt': inpt,
            'target': target,
            'output_in': output_in,
            'output': output,
            'loss': loss
        })

        return exprs
