# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from ..util import ParameterSet, Model, lookup
from ..component import transfer, distance


class MultilayerPerceptron(Model):

    def __init__(self, n_inpt, n_hidden, n_output, 
                 hidden_func, output_func, loss, 
                 inpt=None):
        self.n_inpt = n_inpt
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.hidden_func = hidden_func
        self.output_func = output_func
        self.loss = loss

        pars = self.make_pars()
        exprs = self.make_exprs(inpt, pars)

        super(MultilayerPerceptron, self).__init__(exprs, pars)

    def make_pars(self):
        return ParameterSet(
            inpt_to_hidden=(self.n_inpt, self.n_hidden),
            hidden_bias=self.n_hidden,
            hidden_to_output=(self.n_hidden, self.n_output),
            output_bias=self.n_output)

    def make_exprs(self, inpt, pars):
        inpt = T.matrix('inpt') if inpt is None else inpt
        target = T.matrix('target')

        transfer_hidden = lookup(self.hidden_func, transfer)
        transfer_output = lookup(self.output_func, transfer)
        make_loss = lookup(self.loss, distance)

        hidden_in = T.dot(inpt, pars.inpt_to_hidden) + pars.hidden_bias
        hidden = transfer_hidden(hidden_in)

        output_in = T.dot(hidden, pars.hidden_to_output) + pars.output_bias
        output = transfer_output(output_in)

        loss_rowwise = make_loss(target, output, axis=1)
        loss = loss_rowwise.mean()

        return {
            'inpt': inpt, 
            'target': target,
            'hidden-in': hidden_in,
            'hidden': hidden,
            'output-in': output_in,
            'output': output,
            'loss-rowwise': loss_rowwise,
            'loss': loss
        }
