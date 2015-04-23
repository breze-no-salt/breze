# -*- coding: utf-8 -*-

"""Module that allows the stacking of components."""


import theano.tensor as T

from breze.arch import component
from breze.arch.component.varprop import transfer as vptransfer
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


def make_std(std):
    return (std ** 2 + 1e-8) ** 0.5


class VarpropAffineNonLinear(AffineNonlinear):

    def spec(self):
        spec = {}
        other_spec = super(VarpropAffineNonLinear, self).spec()
        for name, shape in other_spec.items():
            spec[name] = {
                'mean': shape,
                'std': shape
            }
        return spec

    def _init_exprs(self, inpt_mean, inpt_var):
        P = self.parameters
        wm, ws = P.weights.mean, make_std(P.weights.std)
        bm, bs = P.bias.mean, make_std(P.bias.std)

        pres_mean = T.dot(inpt_mean, wm) + bm
        pres_var = (T.dot(inpt_mean ** 2, ws ** 2)
                    + T.dot(inpt_var, wm ** 2)
                    + T.dot(inpt_var, ws ** 2)
                    + bs ** 2)

        f_transfer = lookup(self.transfer, vptransfer)
        post_mean, post_var = f_transfer(pres_mean, pres_var)

        E = self.exprs = get_named_variables(locals())
        self.output = [post_mean, post_var]


class AugmentVariance(Stackable):

    def __init__(self, name, vari=1e-16):
        self.name = name
        self.vari = vari

    def _init_exprs(self, inpt):
        vari = T.zeros_like(inpt) + self.vari
        E = self.exprs = get_named_variables(locals())
        self.output = [inpt, vari]


class DiscardVariance(Stackable):

    def __init__(self, name):
        self.name = name

    def _init_exprs(self, mean, var):
        self.exprs = {'mean': mean}
        self.output = mean,


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












