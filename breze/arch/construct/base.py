# -*- coding: utf-8 -*-


import itertools

from breze.arch.util import ParameterSet, Model
from breze.learn.base import SupervisedBrezeWrapperBase


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

    def __call__(self, *inpt):
        self.forward(*inpt)
        return self.output


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
        spec = self.spec()
        self.parameters = ParameterSet(**spec)

        inpt = inpt,
        for i in self.layers:
            if isinstance(i, Layer):
                i.parameters = getattr(self.parameters, i.name)
                inpt = i(*inpt)
                E[i.name] = i.exprs
            else:
                inpt = i(*inpt)
                if not isinstance(inpt, (list, tuple)):
                    inpt = [inpt]

        if len(inpt) != 1:
            raise ValueError('last layer of stack may only have one output')

        self.output = E['output'] = inpt[0]


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
