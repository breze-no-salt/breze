# -*- coding: utf-8 -*-


import collections

import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda
import theano.misc.gnumpy_utils as gput


def flatten(nested):
    """Flatten nested tuples and/or lists into a flat list."""
    if isinstance(nested, (tuple, list)):
        flat = []
        for elem in nested:
            flat.extend(flatten(elem))
        return flat
    else:
        return [nested]


def unflatten(tmpl, flat):
    """Nest the items in flat into the shape of tmpl."""
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
    lists and/or tuples.

    All other arguments are passed to theano.function without modification."""

    flat_variables = flatten(variables)
    flat_exprs = flatten(exprs)

    flat_function = theano.function(flat_variables, flat_exprs, *args, **kwargs)

    def wrapper(*fargs):
        flat_fargs = flatten(fargs)
        flat_result = flat_function(*flat_fargs)
        result = unflatten(exprs, flat_result)
        return result

    # Expose this to the outside so that fields of theano can be accessed, eg
    # for debug or graph information.
    wrapper.flat_function = flat_function

    return wrapper


def cpu_tensor_to_gpu(tensor):
    """Given a tensor for the CPU return a tensor of the same type and name for
    the GPU."""
    if tensor.ndim == 0:
        result = theano.sandbox.cuda.fscalar(tensor.name)
    elif tensor.ndim == 1:
        result = theano.sandbox.cuda.fvector(tensor.name)
    elif tensor.ndim == 2:
        result = theano.sandbox.cuda.fmatrix(tensor.name)
    elif tensor.ndim == 3:
        result = theano.sandbox.cuda.ftensor3(tensor.name)
    elif tensor.ndim == 4:
        result = theano.sandbox.cuda.ftensor4(tensor.name)
    else:
        raise ValueError('only up to dimension 4')

    return result


def cpu_tensor_to_gpu_nested(inpts, cache=None):
    """Given a list (of lists of...) CPU tensor variables return as list of the
    same types of corresponding GPU tensor varaibles.

    Also return a dictionary containing all substitutions done. This can
    be provided to future calls to not make conversions multiple times.
    """
    if cache is None:
        cache = {}
    inpts_flat = flatten(inpts)
    inpts_flat_conv = []
    for inpt in inpts_flat:
        if inpt in cache:
            item = cache[inpt]
        else:
            item = cpu_tensor_to_gpu(inpt)
            cache[inpt] = item
        inpts_flat_conv.append(item)

    return unflatten(inpts, inpts_flat_conv), cache


def cpu_expr_to_gpu(expr, unsafe=False):
    """Given a CPU expr return the same expression for the GPU.

    If unsafe is set to True, subsequent function calls evaluating the
    expression might return arrays pointing at the same memory region.
    """
    return theano.Out(theano.sandbox.cuda.basic_ops.gpu_from_host(expr),
                      borrow=unsafe)


def cpu_expr_to_gpu_nested(inpts, unsafe=False):
    """Given a list (of lists of...) expressions, return expressions for the
    GPU.

    If unsafe is set to True, subsequent function calls evaluating the
    expression might return arrays pointing at the same memory region.
    """
    inpts_flat = flatten(inpts)
    inpts_flat = [cpu_expr_to_gpu(i, unsafe) for i in inpts_flat]
    return unflatten(inpts, inpts_flat)


def garray_to_cudandarray_nested(lst):
    lst_flat = flatten(lst)
    lst_flat = [gput.garray_to_cudandarray(i) for i in lst_flat]
    lst = unflatten(lst, lst_flat)
    return lst


def cudandarray_to_garray_nested(lst):
    lst_flat = flatten(lst)
    lst_flat = [gput.cudandarray_to_garray(i) for i in lst_flat]
    lst = unflatten(lst, lst_flat)
    return lst


def gnumpy_func_wrap(f):
    """Wrap a function that accepts and returns CudaNdArrays to accept and
    return gnumpy arrays."""
    def inner(*args):
        args = garray_to_cudandarray_nested(args)
        res = f(*args)
        print type(res)
        if isinstance(res, list):
            res = cudandarray_to_garray_nested(res)
        else:
            res = gput.cudandarray_to_garray(res)
        return res
    return inner


def lookup(what, where, default=None):
    """Return ``where.what`` if what is a string, otherwise what. If not found
    return ``default``."""
    if isinstance(what, (str, unicode)):
        res = getattr(where, what, default)
    else:
        res = what
    return res


def lookup_some_key(what, where, default=None):
    """Given a list of keys ``what``, return the first of those to which there
    is an item in ``where``.

    If nothing is found, return ``default``.
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
    """ParameterSet class.

    This class provides functionality to group several Theano tensors of
    different sizes in a consecutive chunk of memory. The main aim of this is
    to allow a view on several tensors as a single long vector.

    In the following, a (parameter) array refers to a concrete instantiation of
    a parameter variable (with concrete values) while a (parameter)
    tensor/variable refers to the symbolic Theano variable.


    Parameters
    ----------

    Initialization takes a variable amount of keyword arguments, where each has
    to be a single integer or a tuple of arbitrary length containing only
    integers. For each of the keyword argument keys a tensor of the shape given
    by the value will be created. The key is the identifier of that variable.


    Attributes
    ----------

    n_pars : integer
        Total amount of parameters.

    flat : Theano vector
        Flat one dimensional tensor containing all the different tensors
        flattened out. Symbolic pendant to ``data``.

    data : array_like
        Concrete array containig all the different arrays flattened out.
        Concrete pendant to ``flat``.

    views : dictionary
        All parameter arrays can be accessed by with their identifier as key
        in this dictionary.

    All symbolic variables can be accessed as attributes of the object, all
    concrete variables as keys. E.g. parameter_set.x references the symbolic
    variable, while parameter_set['x'] will give you the concrete array.
    """

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
            region = region.reshape(shape)
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
    """Model class.

    Intended as a base class for parameterized models providing a convenience
    method for compilation and a common interface.

    We partition Theano variables for parametrized models in three groups.
    (1) The *adaptable parameters*, (2) *external variables* such as inputs and
    targets, the data (3) *expressions* composed out of the two, such as the
    prediction of a model or the loss resulting from those.

    Attributes
    ----------

    pars : ParameterSet object
        Holding the adaptable parameters of the object.

    exprs : dictionary
        Containig the expressions. Out of convenience, the external variables
        are held in here as well.

    updates : dictionary containing update variables, e.g. due to the use of
        ``theano.scan``.


    Expression Names
    ----------------

    There are several "reserved" names for expressions.

      - ``inpt``: observations of a supervised or unsupervised model,
      - ``target``: desired outputs of a supervised model,
      - ``loss``: quantity to be optimized for fitting the parameters;
        might not refer to the criterion of interest, but instead to a
        regularzied objective.
      - ``true_loss``: Quantity of interst for the user, e.g. the loss without
        regularization or the empirical risk.

    Overriding these names is possible in general, but is part of the interface
    and will lead to unexpected behaviour with functionality building upon this.
    """

    def __init__(self):
        self.updates = collections.defaultdict(dict)
        self.init_pars()
        self.init_exprs()
        self.gpu_variable_cache = None

    def init_pars(self):
        pass

    def init_exprs(self):
        pass

    def function(self, variables, exprs, mode=None, explicit_pars=False,
                 givens=None,
                 on_unused_input='raise'):
        """Return a compiled function for the given `exprs` given `variables`.


        Parameters
        ----------

        variables : list of strings
            Each string refers to an item in ``.exprs`` and is considered an
            input to the function.

        exprs : (List of) Theano expression or string
            Expressions for which to create the function. If a single expression
            is given, the function will return a single value; if a list is
            given, the result will be a tuple containing one element for each.
            An expression can either be a Theano expression or a string. In the
            latter case, the corresponding expression will be retrieved from
            ``.exprs``.

        mode : string or None, optional, default: None
            Mode to use for compilation. Passed on to ``theano.function``.
            See Theano documentation for details.

        explicit_pars: boolean, optional, default: False
            If True, the first argument to the function is expected to be an
            array representing the adaptable parameters of the model.

        givens : dictionary, optional, default: None
            Dictionary of substitutions for compilation. Not passed on to
            ``theano.function``, instead the expressions are cloned. See code
            for further details.

        on_unused_input: string
            Specifiy behaviour in case of unused inputs. Passed on to
            ``theano.function``. See Theano documentation for details.
        """

        def lookup(varname):
            res = getattr(self.parameters, varname, None)
            if res is None:
                res = self.exprs[i]
            return res
        variables = [lookup(i) if isinstance(i, str) else i
                     for i in variables]

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

        # We need to clone instead of using the givens parameter of
        # theano.function, because otherwise we might get an theano error
        # with conflicting replacements. (See theano/compile/pfunc.py:162,
        # rebuild_collect_shared.)
        if givens is not None:
            if isinstance(exprs, list):
                exprs = [theano.clone(e, givens) for e in exprs]
            else:
                exprs = theano.clone(exprs, givens)

        if explicit_pars:
            pars = T.dvector(self.parameters.flat.name + '-substitute')
            variables = [pars] + variables
            givens = {}
            givens[self.parameters.flat] = pars

        # Build update dictionary.
        updates = collections.defaultdict(lambda: {})
        if isinstance(exprs, (list, tuple)):
            flat_exprs = flatten(exprs)
            for expr in flat_exprs:
                # TODO: last takes all, maybe should throw an error.
                updates.update(self.updates[expr])
        else:
            updates.update(self.updates[exprs])

        if theano.config.device == 'gpu':
            variables, self.gpu_variable_cache = cpu_tensor_to_gpu_nested(
                variables, self.gpu_variable_cache
            exprs = cpu_expr_to_gpu_nested(exprs)

        f = theano_function_with_nested_exprs(
            variables, exprs, givens=givens, mode=mode,
            on_unused_input=on_unused_input, updates=updates)

        if theano.config.device == 'gpu':
            f = gnumpy_func_wrap(f)

        return f


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
