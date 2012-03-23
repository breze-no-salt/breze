# -*- coding: utf-8 -*-


import contextlib
import sys
import time

import numpy as np
import theano.tensor as T
import theano


def lookup(what, where):
    """Return where.what if what is a string, else what."""
    return getattr(where, what) if isinstance(what, (str, unicode)) else what


class ParameterSet(object):

    def __init__(self, **kwargs):
        # Make sure all size specifications are tuples.
        kwargs = dict((k, v if isinstance(v, tuple) else (v,))
                      for k, v in kwargs.iteritems())
        # Find out total size of needed parameters and create memory for it.
        sizes = [np.prod(i) for i in kwargs.values()]
        self.n_pars = sum(sizes)
        self.flat = theano.shared(np.empty(self.n_pars), name='flat')
        # Go through parameters and assign space and variable.
        self.views = {}
        n_used = 0 	# Number of used parameters.
        self.data = self.flat.get_value(borrow=True, return_internal_type=True)
        for (key, shape), size in zip(kwargs.items(), sizes):
            # Make sure the key is legit -- that it does not overwrite anything.
            if hasattr(self, key):
                raise ValueError("%s is an illegal name for a variable")
  
            # Get the region from the big flat array.
            region = self.data[n_used:n_used + size]
            # Then shape it correctly and make it accessible from the outside.
            region.shape = shape
            self.views[key] = region
  
            # Get the right variable as a subtensor.
            var = self.flat[n_used:n_used + size].reshape(shape)
            var.name = key
            setattr(self, key, var)
  
            n_used += size

    def __contains__(self, key):
        return key in self.views
  
    def __getitem__(self, key):
        return self.views[key]

    def __setitem__(self, key, value):
        self.views[key][:] = value


class Model(object):

    def __init__(self):
        self.init_pars()
        self.init_exprs()

    def function(self, variables, exprs, mode=None, explicit_pars=False):
        """Return a function for the given `exprs` given `variables`.

        If `mode` is different to None, the specified mode is used, otherwise
        the Theano default. In case `explicit_pars` is set to True, the first
        argument of the function needs to be a numpy array from which the
        parameters of the LossBased model will be extracted."""
        def lookup(varname):
            res = getattr(self.parameters, varname, None)
            if res is None:
                res = self.exprs[i]
            return res
        variables = [lookup(i) if isinstance(i, str) else i
                     for i in variables]

        if mode is None:
            mode = theano.Mode(linker='cvm')

        if isinstance(exprs, (str, unicode)):
            # We are only being given a single string expression.
            exprs = self.exprs[exprs]
        elif isinstance(exprs, theano.tensor.basic.TensorVariable):
            exprs = exprs
        else:
            # We have several, either string or variable, thus make it a list
            # and substitute the strings.
            exprs = list(exprs)
            exprs = [self.exprs[i] if isinstance(i, str) else i for i in exprs]

        if explicit_pars:
            pars = T.dvector(self.parameters.flat.name + '-substitute')
            variables = [pars] + variables
            givens = [(self.parameters.flat, pars)]
        else:
            givens = []

        return theano.function(variables, exprs, givens=givens, mode=mode)
