# -*- coding: utf-8 -*-


import itertools

from breze.arch.util import ParameterSet


def invalid_declare(*args, **kwargs):
    raise ValueError('declare cannot be called anymore since '
                     'Layer ws pickled or copied.')


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

    def __getstate__(self):
        # The following makes sure that the object can be pickled by replacing
        # the .declare method.
        #
        # Why is it ok to do so? If we pickle a Layer, we can expect it to be
        # already finalized, i.e. _forward has been called. This is being done
        # during initialisation, which means we will not need declare anymore
        # anyway.
        state = self.__dict__.copy()
        state['declare'] = invalid_declare
        return state
