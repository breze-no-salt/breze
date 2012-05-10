# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T

from ..util import ParameterSet, Model, lookup
from ..component import transfer, distance


class Linear(Model):

    def __init__(self, n_inpt, n_output, out_transfer, loss):
        self.n_inpt = n_inpt
        self.n_output = n_output
        self.out_transfer = out_transfer
        self.loss = loss

        self.init_pars()
        self.init_exprs()

    def init_pars(self):
        parspec = self.get_parameter_spec(self.n_inpt, self.n_output)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        self.exprs = self.make_exprs(
            T.matrix('inpt'), self.parameters.in_to_out, self.parameters.bias,
            self.out_transfer, self.loss)

    @staticmethod
    def get_parameter_spec(n_inpt, n_output):
        return {'in_to_out': (n_inpt, n_output),
                'bias': n_output}

    @staticmethod
    def make_exprs(inpt, in_to_out, bias, out_transfer, loss):
        f_out = lookup(out_transfer, transfer)
        f_loss = lookup(loss, distance)

        target = T.matrix('target')

        output_in = T.dot(inpt, in_to_out) + bias
        output = f_out(output_in)

        loss_rowwise = f_loss(target, output, axis=1)
        loss = loss_rowwise.mean()

        return {
            'inpt': inpt,
            'target': target,
            'output_in': output_in,
            'output': output,
            'loss_rowwise': loss_rowwise,
            'loss': loss
        }
