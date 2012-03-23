# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from ..component import distance
from linear import Linear

class RIM_LR(Linear):
    def __init__(self, n_inpt, n_output, c_rim):
        self.c_rim = c_rim
        super(RIM_LR, self).__init__(n_inpt=n_inpt,
                n_output=n_output, out_transfer='softmax',
                loss='cross_entropy' )

    def init_exprs(self):
        self.exprs = self.make_exprs(T.matrix('inpt'), 
                self.parameters.in_to_out, self.parameters.bias,
                self.out_transfer, self.loss, self.c_rim)

    @staticmethod
    def make_exprs(inpt, in_to_out, bias, out_transfer, loss, c_rmi):
        exprs = Linear.make_exprs(inpt, in_to_out, bias, out_transfer, loss)
        output = exprs['output']

        marginal = output.mean(axis=0)
        cond_entropy = distance.discrete_entropy(output, axis=1).mean()
        entropy = distance.discrete_entropy(marginal)

        # negative mutual information -> we are minimizing
        neg_mi = cond_entropy - entropy
        l2 = (in_to_out**2).sum()

        exprs['neg_mi'] = neg_mi
        exprs['l2'] = l2

        exprs['loss_reg'] = neg_mi + c_rmi * l2
        
        return exprs
