

import numpy as np
import theano.tensor as T


from ..util import ParameterSet, Model, lookup
from ..component import transfer, distance




class LogisticRegression(Model):

    def __init__(self, n_inpt, n_output, transfer, loss, inpt=None):
        self.n_inpt = n_inpt
        self.n_output = n_output
        self.transfer = transfer
        
        self.loss = loss

        pars = self.make_pars()
        exprs = self.make_exprs(inpt, pars)

        super(LogisticRegression, self).__init__(exprs, pars)

    def make_pars(self):
        return ParameterSet(weights=(self.n_inpt, self.n_output),
                bias=self.n_output)

    def make_exprs(self, inpt, pars):
        inpt = T.matrix('inpt') if inpt is None else inpt
        target = T.matrix('target')

        make_loss = lookup(self.loss, distance)
        transfer_func = lookup(self.transfer, transfer)

        output_in = T.dot(inpt, pars.weights) + pars.bias
        output = transfer_func(output_in)

        loss_rowwise = make_loss(target, output, axis=1)
        loss = loss_rowwise.mean()

        return {
            'inpt': inpt,
            'target': target,
            'output-in': output_in,
            'output': output,
            'loss-rowwise': loss_rowwise,
            'loss': loss
            }
