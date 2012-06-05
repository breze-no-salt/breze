# -*- coding: utf-8 -*-


import collections
import contextlib
import sys
import time

import numpy as np
import theano.tensor as T
import theano

def flatten(nested):
    """Flattens nested tuples and/or lists into a flat list"""
    if isinstance(nested, (tuple, list)):
        flat = []
        for elem in nested:
            flat.extend(flatten(elem))
        return flat
    else:
        return [nested]

def unflatten(tmpl, flat):
    """Nests the items in flat into the shape of tmpl"""
    def unflatten_recursive(tmpl, flat):
        if isinstance(tmpl, (tuple, list)):
            nested = []
            for sub_tmpl in tmpl:
                sub_nested, flat = unflatten_recursive(sub_tmpl, flat)
                nested.append(sub_nested)
            if isinstance(tmpl, tuple):
                nested = tuple(nested)
            return nested, flat
        else:
            return flat[0], flat[1:]

    nested, _ = unflatten_recursive(tmpl, flat)
    return nested


def theano_function_with_nested_exprs(variables, exprs, *args, **kwargs):
    """Creates and returns a theano.function that takes values for `variables` 
    as arguments, where `variables` may contain nested lists and/or tuples, 
    and returns values for `exprs`, where again `exprs` may contain nested
    lists and/or tuples. All other arguments are passed to theano.function 
    without modification."""

    flat_variables = flatten(variables)
    flat_exprs = flatten(exprs)

    flat_function = theano.function(flat_variables, flat_exprs, *args, **kwargs)

    def wrapper(*fargs):
        flat_fargs = flatten(fargs)
        flat_result = flat_function(*flat_fargs)
        result = unflatten(exprs, flat_result)
        return result

    return wrapper


def lookup(what, where):
    """Return where.what if what is a string, else what."""
    return getattr(where, what) if isinstance(what, (str, unicode)) else what


def lookup_some_key(what, where, default=None):
    """Return where[w] where w is the first element in `what` which `where` has.
    """
    for w in what:
        try:
            return where[w]
        except KeyError:
            pass
    return default


def opt_from_model(model, fargs, args, opt_klass, opt_kwargs):
    """Return an optimizer object given a model and an optimizer specification.
    """
    d_loss_d_pars = T.grad(model.exprs['loss'], model.parameters.flat)
    f = model.function(fargs, 'loss', explicit_pars=True)
    fprime = model.function(fargs, d_loss_d_pars, explicit_pars=True)
    opt = opt_klass(model.parameters.data, f, fprime, args=args, **opt_kwargs)
    return opt


class ParameterSet(object):

    def __init__(self, **kwargs):
        # Make sure all size specifications are tuples.
        kwargs = dict((k, v if isinstance(v, tuple) else (v,))
                      for k, v in kwargs.iteritems())
        # Find out total size of needed parameters and create memory for it.
        sizes = [np.prod(i) for i in kwargs.values()]
        self.n_pars = sum(sizes)
        self.flat = theano.shared(np.empty(self.n_pars, dtype=theano.config.floatX), name='flat')
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
        self.updates = collections.defaultdict(lambda: {})
        self.init_pars()
        self.init_exprs()

    def init_pars(self):
        pass

    def init_exprs(self):
        pass

    def __getstate__(self):
        dct = self.__dict__.copy()
        dct['updates'] = dict(dct['updates'])
        return dct

    def __setstate__(self, state):
        dct = state['updates'] 
        state['updates'] = collections.defaultdict(lambda: {})
        state['updates'].update(dct)
        self.__dict__.update(state)

    def function(self, variables, exprs, mode=None, explicit_pars=False,
                 on_unused_input='raise'):
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

        # Build update dictionary.
        updates = collections.defaultdict(lambda: {})        
        if isinstance(exprs, (list, tuple)):
            flat_exprs = flatten(exprs)
            for expr in flat_exprs:
                # TODO: last takes all, maybe should throw an error.
                updates.update(self.updates[expr])
        else:
            updates.update(self.updates[exprs])

        return theano_function_with_nested_exprs(variables, exprs, 
                                                 givens=givens, 
                                                 mode=mode,
                                                 on_unused_input=on_unused_input,
                                                 updates=updates)


class PrintEverythingMode(theano.Mode):
    def __init__(self):
        def print_eval(i, node, fn):
            print '<' * 50
            print i, node, [input[0] for input in fn.inputs],
            fn()
            print [output[0] for output in fn.outputs]
            print '>' * 50
        wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()], [print_eval])
        super(PrintEverythingMode, self).__init__(wrap_linker, optimizer='fast_compile')


class WarnNaNMode(theano.Mode):
    def __init__(self):
        def print_eval(i, node, fn):
            fn()
            for i, inpt in enumerate(fn.inputs):
                try:
                    if np.isnan(inpt[0]).any():
                        print 'nan detected in input %i of %s' % (i, node)
                        import pdb
                        pdb.set_trace()
                except TypeError:
                    print 'could not check for NaN in:', inpt
        wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()], [print_eval])
        super(WarnNaNMode, self).__init__(wrap_linker, optimizer='fast_compile')
