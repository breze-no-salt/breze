# -*- coding: utf-8 -*-

"""Module that allows the stacking of components."""


import theano.tensor as T

from breze.arch import component
import component.transfer, component.loss
from util import ParameterSet, Model, lookup, get_named_variables


class Stackable(object):

    def spec(self):
        return {}


class AffineNonlinear(Stackable):

    @property
    def n_inpt(self):
        return self._n_inpt

    @property
    def n_output(self):
        return self._n_output

    def __init__(self, n_inpt, n_output, transfer='identity', bias=True):
        self._n_inpt = n_inpt
        self._n_output = n_output
        self.transfer = transfer
        self.bias = True

    def spec(self):
        spec = {
            'weights': (self.n_inpt, self.n_output)
        }
        if self.bias:
            spec['bias'] = self.n_output,
        return spec

    def _init_exprs(self, *inpt):
        inpt, = inpt
        P = self.parameters

        output_pre_transfer = T.dot(inpt, P.weights)
        if self.bias:
            output_pre_transfer += P.bias

        f_transfer = lookup(self.transfer, component.transfer)
        output = f_transfer(output_pre_transfer)

        output_pre_transfer.name = 'output_pre_transfer'
        output.name = 'output'

        E = self.exprs = get_named_variables(locals())
        self.output = [output]


class SupervisedLoss(Stackable):

    def __init__(self, loss, target, comp_dim=1):
        self.loss = loss
        self.target = target
        self.comp_dim = 1

    def _init_exprs(self, *inpt):
        inpt, = inpt
        f_loss = lookup(self.loss, component.loss)

        coord_wise = f_loss(self.target, inpt)
        sample_wise = coord_wise.sum(self.comp_dim)
        total = sample_wise.mean()

        E = self.exprs = get_named_variables(locals())


class Stack(Model):

    def __init__(self):
        self.layers = []
        super(Stack, self).__init__()

    def spec(self):
        return dict((i.name, i.spec()) for i in self.layers)

    def finalize(self, *inpt):
        E = self.exprs = {'inpt': inpt}
        spec = self.spec()
        self.parameters = ParameterSet(**spec)

        self.layers[0].parameters = getattr(
            self.parameters, self.layers[0].name)
        self.layers[0]._init_exprs(*inpt)
        for incoming, outgoing in zip(self.layers[:-1], self.layers[1:]):
            outgoing.parameters = getattr(self.parameters, outgoing.name)
            outgoing._init_exprs(*incoming.output)



        E['output'] = inpt












