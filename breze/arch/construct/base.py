# -*- coding: utf-8 -*-


import itertools

import numpy as np
import theano
import theano.tensor as T

from breze.arch.util import ParameterSet


class Layer(object):

    _counter = itertools.count()

    def __init__(self, declare=None, name=None):
        self.make_name(name)

        if declare is None:
            self.parameters = ParameterSet()
            self.declare = self.parameters.declare
        else:
            self.declare = declare

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
