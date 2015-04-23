# -*- coding: utf-8 -*-


import itertools

import theano.tensor as T

from breze.arch.component import transfer as _transfer, loss as _loss
from breze.arch.util import ParameterSet, Model, lookup, get_named_variables
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

    def spec(self):
        return {}

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


class Stack(Model, Layer):

    def __init__(self, layers, name=None):
        self.layers = layers
        Model.__init__(self)
        Layer.__init__(self, name)

    def spec(self):
        return dict((i.name, i.spec()) for i in self.layers)

    def forward(self, inpt):
        Layer.forward(self, inpt)

        # First part: predictive model.
        E = self.exprs = {'inpt': inpt}
        spec = self.spec()
        self.parameters = ParameterSet(**spec)

        inpt = inpt,
        for i in self.layers:
            i.parameters = getattr(self.parameters, i.name)
            i.forward(*inpt)
            E[i.name] = i.exprs

            inpt = i.output

        assert len(i.output) == 1, 'last layer of stack may only have one output'

        self.output = E['output'] = i.output[0]


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

        self.loss.forward(self.output)
        self.exprs['loss'] = self.loss.exprs['total']
        self.exprs['target'] = self.loss.exprs['target']
