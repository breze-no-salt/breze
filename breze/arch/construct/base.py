# -*- coding: utf-8 -*-


import itertools

import numpy as np
import theano
import theano.tensor as T

from breze.arch.util import ParameterSet, Model
from breze.learn.base import (
    SupervisedBrezeWrapperBase, UnsupervisedBrezeWrapperBase)


class Layer(object):
    """Layer class.

    Base class for all components that perform computations. Layers have several
    properties.

    For one, a Layer object is named. That is, it has an attribute ``.name``
    which is supposed to hold a unique string that identifies the object. This
    is not enforced however.

    Further, they have a method ``.forward(*inpt)``. This method takes a
    variable number of inputs and creates the ``.output`` attribute of the
    object. This is an iterable, in turn holding a variable number of outputs.
    Both are Theano expressions.

    Further, ``.forward`` may only be called once. After that, an error will be
    thrown.

    A shorthand is to just use ``__call__()``, which will return the output
    expressions. The use of this is that we can use arbitrary functions that
    receive Theano variables and return Theano variables as layers.

    Layer can be parameterised, i.e. have adaptable parameters that an outer
    learning algorithm can tune. For that, the ``.spec()`` method is supposed to
    return a dictionary mapping names to tuples specifying the shape of the
    parameters. The dictionary can be recursive, i.e. each of its items can be
    a dictionary itself. The leafs of the so formed tree need to be tuples.
    Integers are turned into a single element tuple and are valid as well.
    """

    _counter = itertools.count()

    def __init__(self, name=None):
        self.make_name(name)
        self._forwarded = False
        self._spec = {}
        self._parameterized = {}

    def spec(self):
        return self._spec

    def make_name(self, name):
        """Give the layer a unique name.

        If ``name`` is None, construct a name of the form 'N-#' where N is the
        class name and # is a global counter to avoid collisions.
        """
        if name is None:
            self.name = '%s-%i' % (
                self.__class__.__name__, self._counter.next())
        else:
            self.name = name

    def forward(self, *inpt):
        if self._forwarded:
            raise ValueError('already forwarded')

        self._forwarded = True

    def __call__(self, *inpt):
        self.forward(*inpt)
        return self.output

    def parameterized(self, name, shape):
        # Theano replaces tensors with shape (1, x) or (x, 1) with T.col and
        # T.row repsectively.
        if len(shape) == 2:
            x = T.matrix()
            if shape == (1, 1):
                x = T.TensorType(theano.config.floatX, (True,))()
            elif shape[1] == 1:
                x = T.col()
            elif shape[0] == 1:
                x = T.row()

        if len(shape) == 1:
            x = T.vector()
            if shape[0] == 1:
                x = T.TensorType(theano.config.floatX, (True,))()

        x.tag.test_value = np.zeros(shape).astype(theano.config.floatX)

        self._spec.update({name: shape})
        self._parameterized.update({name: x})
        return x


class Stack(Model, Layer):

    def __init__(self, layers, name=None):
        self.layers = layers
        Model.__init__(self)
        Layer.__init__(self, name)

    def spec(self):
        return dict((i.name, i.spec())
                    for i in self.layers
                    if hasattr(i, 'spec'))

    def forward(self, inpt):
        Layer.forward(self, inpt)

        # First part: predictive model.
        E = self.exprs = {'inpt': inpt}

        inpt = inpt,
        for i in self.layers:
            if isinstance(i, Layer):
                inpt = i(*inpt)
                E[i.name] = i.exprs
            else:
                inpt = i(*inpt)
                if not isinstance(inpt, (list, tuple)):
                    inpt = [inpt]

        if len(inpt) != 1:
            raise ValueError('last layer of stack may only have one output')

        self.output = E['output'] = inpt[0]

    def _replace_param_dummies(self):
        spec = self.spec()
        self.parameters = ParameterSet(**spec)

        def recursive_replace(exprs, replaceby):
            for i in exprs:
                if isinstance(exprs[i], dict):
                    recursive_replace(exprs[i], replaceby)
                else:
                    exprs[i] = theano.clone(exprs[i], replaceby)

        replaceby = {}
        for i in self.layers:
            if isinstance(i, Layer):
                for p in i._parameterized:
                    up = {i._parameterized[p]:
                          getattr(getattr(self.parameters, i.name), p)}
                    replaceby.update(up)

        recursive_replace(self.exprs, replaceby)


class SupervisedStack(Stack, SupervisedBrezeWrapperBase):

    def __init__(self, layers, loss, name=None):
        self.loss = loss
        super(SupervisedStack, self).__init__(
            layers, name)

    def predict(self, X):
        if getattr(self, 'f_predict', None) is None:
            self.f_predict = self.function(['inpt'], 'output')
        return self.f_predict(X)

    def forward(self, inpt):
        super(SupervisedStack, self).forward(inpt)

        self.loss(self.output)
        self.exprs['loss'] = self.loss.exprs['total']
        self.exprs['target'] = self.loss.exprs['target']
        if 'imp_weight' in self.loss.exprs:
            self.exprs['imp_weight'] = self.loss.exprs['imp_weight']


class UnsupervisedStack(Stack, UnsupervisedBrezeWrapperBase):

    def __init__(self, layers, loss, name=None):
        self.loss = loss
        super(UnsupervisedStack, self).__init__(layers, name)

    def forward(self, inpt):
        super(UnsupervisedStack, self).forward(inpt)
        self.loss(self.output)
        self.exprs['loss'] = self.loss.exprs['total']
        if 'imp_weight' in self.loss.exprs:
            self.exprs['imp_weight'] = self.loss.exprs['imp_weight']
