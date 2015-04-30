# -*- coding: utf-8 -*-


import itertools

import numpy as np
import theano
import theano.tensor as T

from breze.arch.util import ParameterSet


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

    def __init__(self, declare=None, name=None):
        self.make_name(name)

        if declare is None:
            self.parameters = ParameterSet()
            self._declare = self.parameters.declare
        else:
            self._declare = declare

        self._forward()

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
