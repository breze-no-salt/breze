# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from ..util import ParameterSet, Model, lookup
from ..component import transfer, distance


class MultilayerPerceptron(Model):

    def __init__(self, n_inpt, n_hidden, n_output, 
                 hidden_transfer, output_transfer, loss):
        self.n_inpt = n_inpt
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.hidden_transfer = hidden_transfer
        self.output_transfer = output_transfer
        self.loss = loss

        self.init_pars()
        self.init_exprs()

    def init_pars(self):
        self.parameters = ParameterSet(
            in_to_hidden=(self.n_inpt, self.n_hidden),
            hidden_to_out=(self.n_hidden, self.n_output),
            hidden_bias=self.n_hidden,
            out_bias=self.n_inpt)

    def init_exprs(self):
        self.exprs = self.make_exprs(
            T.matrix('inpt'), T.matrix('target'),
            self.parameters.in_to_hidden, self.parameters.hidden_to_out,
            self.parameters.hidden_bias, self.parameters.out_bias,
            self.hidden_transfer, self.output_transfer, self.loss)

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

        loss_rowwise = f_loss(target, output, axis=1)
        loss = loss_rowwise.mean()

        return {
            'inpt': inpt, 
            'target': target,
            'hidden_in': hidden_in,
            'hidden': hidden,
            'output_in': output_in,
            'output': output,
            'loss_rowwise': loss_rowwise,
            'loss': loss
        }
