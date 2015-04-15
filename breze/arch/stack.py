# -*- coding: utf-8 -*-

"""Module that allows the stacking of components."""


import theano.tensor as T

from breze.arch import component
import component.transfer, component.loss
from util import ParameterSet, Model, lookup, get_named_variables
from breze.learn.base import (
    SupervisedBrezeWrapperBase, UnsupervisedBrezeWrapperBase)


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

    def __init__(self, name, n_inpt, n_output, transfer='identity', bias=True):
        self.name = name
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

        E = self.exprs = get_named_variables(locals())
        self.output = [output]


class SupervisedLoss(Stackable):

    def __init__(self, loss, target_class=T.matrix, comp_dim=1):
        self.loss = loss
        self.target = target_class('target')
        self.comp_dim = 1

    def _init_exprs(self, *inpt):
        inpt, = inpt
        f_loss = lookup(self.loss, component.loss)

        coord_wise = f_loss(self.target, inpt)
        sample_wise = coord_wise.sum(self.comp_dim)
        total = sample_wise.mean()

        E = self.exprs = get_named_variables(locals())
        E['target'] = self.target



class Stack(Model):

    def __init__(self, inpt_class=T.matrix):
        self.inpt_class = inpt_class

        self.layers = []
        self.loss = None
        self._finalized = False
        super(Stack, self).__init__()

    def spec(self):
        return dict((i.name, i.spec()) for i in self.layers)

    def finalize(self):
        if self._finalized:
            raise ValueError('already finalized')

        if self.loss is None:
            raise ValueError('no loss specified')

        # First part: predictive model.
        inpt = self.inpt_class('inpt')
        E = self.exprs = {'inpt': inpt}
        spec = self.spec()
        self.parameters = ParameterSet(**spec)

        inpt = inpt,
        for i in self.layers:
            i.parameters = getattr(self.parameters, i.name)
            i._init_exprs(*inpt)
            E[i.name] = i.exprs

            inpt = i.output

        E['output'] = i.output

        # Second part: loss function.
        self.loss._init_exprs(i.output)
        self.exprs['loss'] = self.loss.exprs['total']

        self._finalized = True


class SupervisedStack(Stack, SupervisedBrezeWrapperBase):

    def predict(self, X):
        if getattr(self, 'f_predict', None) is None:
            self.f_predict = self.function(['inpt'], 'output')
        return self.f_predict(X)

    def finalize(self):
        super(SupervisedStack, self).finalize()
        self.exprs['target'] = self.loss.exprs['target']












