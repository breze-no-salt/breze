# -*- coding: utf-8 -*-

import os
import sys
import collections
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda
import theano.misc.gnumpy_utils as gput


try:
    gpu_environ = os.environ['BREZE_PARAMETERSET_DEVICE']
    if gpu_environ == 'gpu':
        GPU = True
    elif gpu_environ == 'cpu':
        GPU = False
    else:
        print "BREZE_PARAMETERSET_DEVICE must be either 'cpu' or 'gpu'"
        sys.exit(1)
except KeyError:
    GPU = theano.config.device == 'gpu'

if GPU:
    import gnumpy


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
    """Creates and returns a ``theano.function`` that takes values for
    ``variables``
    as arguments, where ``variables` may contain nested lists and/or tuples,
    and returns values for ``exprs``, where again ``exprs`` may contain nested
    lists and/or tuples.

    All other arguments are passed to ``theano.function`` without
    modification."""

    flat_variables = flatten(variables)
    flat_exprs = flatten(exprs)

    flat_function = theano.function(
        flat_variables, flat_exprs,
        *args, **kwargs)

    def wrapper(*fargs):
        flat_fargs = flatten(fargs)
        flat_result = flat_function(*flat_fargs)
        result = unflatten(exprs, flat_result)
        return result

    # Expose this to the outside so that fields of theano can be accessed, eg
    # for debug or graph information.
    wrapper.theano_func = flat_function

    return wrapper


def cpu_tensor_to_gpu(tensor):
    """Given a tensor for the CPU return a tensor of the same type and name for
    the GPU."""
    name = '%s-gpu' % tensor.name
    if tensor.ndim == 0:
        result = theano.sandbox.cuda.fscalar(name)
    elif tensor.ndim == 1:
        result = theano.sandbox.cuda.fvector(name)
    elif tensor.ndim == 2:
        result = theano.sandbox.cuda.fmatrix(name)
    elif tensor.ndim == 3:
        result = theano.sandbox.cuda.ftensor3(name)
    elif tensor.ndim == 4:
        result = theano.sandbox.cuda.ftensor4(name)
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
    if isinstance(expr, theano.sandbox.cuda.var.CudaNdarrayVariable):
        return expr

    expr_ = T.cast(expr, 'float32')
    expr_ = theano.Out(theano.sandbox.cuda.basic_ops.gpu_from_host(expr),
                       borrow=unsafe)

    expr_.name = expr.name
    return expr_


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
        if isinstance(res, list):
            res = cudandarray_to_garray_nested(res)
        else:
            # TODO: check for CudaNdArray instance instead
            if not isinstance(res, (float, np.ndarray)):
                res = gput.cudandarray_to_garray(res)
        return res
    inner.theano_func = f.theano_func
    return inner


def lookup(what, where, default=None):
    """Return ``where.what`` if what is a string, otherwise what. If not found
    return ``default``."""
    if isinstance(what, (str, unicode)):
        res = getattr(where, what, default)
    else:
        res = what
    return res


def get_named_variables(dct, name=True, overwrite=False, prefix=''):
    """Return a dictionary with all the items from ``dct`` with only Theano
    variables/expressions.

    If ``name`` is set to True, the variables will be named accordingly, however
    not be overwritten unless ``overwrite`` is True as well.
    """
    exprs = [('%s%s' % (prefix, k), v) for k, v in dct.items()
             if isinstance(v, theano.tensor.basic.TensorVariable)]

    if name:
        for k, v in exprs:
            if not hasattr(v, 'name') or overwrite:
                v.name = '%s%s' % (prefix, k)
    return dict(exprs)


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


def theano_expr_bfs(expr):
    """Generator function to walk a Theano expression graph in breadth first."""
    stack = [expr]
    marked = set(stack)
    while True:
        if not stack:
            break
        expr = stack.pop()
        candidates = expr.owner.inputs if hasattr(expr.owner, 'inputs') else []
        candidates = [i for i in candidates if i not in marked]

        stack += candidates
        marked |= set(candidates)

        yield expr


def tell_deterministic(expr):
    """Return True iff no random number generator is in the expression graph."""
    return all(not hasattr(i, 'rng') for i in theano_expr_bfs(expr))


class ParameterSet(object):
    """ParameterSet class.

    This class provides functionality to group several Theano tensors of
    different sizes in a consecutive chunk of memory. The main aim of this is
    to allow a view on several tensors as a single long vector.

    In the following, a (parameter) array refers to a concrete instantiation of
    a parameter variable (with concrete values) while a (parameter)
    tensor/variable refers to the symbolic Theano variable.


    Initialization takes a variable amount of keyword arguments, where each has
    to be a single integer or a tuple of arbitrary length containing only
    integers. For each of the keyword argument keys a tensor of the shape given
    by the value will be created. The key is the identifier of that variable.

    All symbolic variables can be accessed as attributes of the object, all
    concrete variables as keys. E.g. parameter_set.x references the symbolic
    variable, while parameter_set['x'] will give you the concrete array.

    Attributes
    ----------

    n_pars : integer
        Total amount of parameters.

    flat : Theano vector
        Flat one dimensional tensor containing all the different tensors flattened out. Symbolic pendant to ``data``.

    data : array_like
        Concrete array containig all the different arrays flattened out.
        Concrete pendant to ``flat``.

    views : dict
        All parameter arrays can be accessed by with their identifier as key
        in this dictionary.
    """

    def __init__(self, **kwargs):
        # Make sure all size specifications are tuples.
        kwargs = dict((k, v if isinstance(v, tuple) else (v,))
                      for k, v in kwargs.iteritems())

        # Find out total size of needed parameters and create memory for it.
        sizes = [np.prod(i) for i in kwargs.values()]

        self.n_pars = sum(sizes)

        # Create two representations of the parameters of the object. The first
        # is the symbolic theano variable (of which the type is GPU/CPU
        # specific), the second either a gnumpy or numpy array (depending on
        # GPU/CPU again). Also set a default size for testing.
        if GPU:
            self.data = gnumpy.zeros(self.n_pars)
            self.flat = theano.sandbox.cuda.fvector('parameters')
        else:
            self.data = np.empty(self.n_pars).astype(theano.config.floatX)
            self.flat = T.vector('parameters')

        self.flat.tag.test_value = self.data

        # Go through parameters and assign space and variable.
        self.views = {}
        n_used = 0 	# Number of used parameters.

        for (key, shape), size in zip(kwargs.items(), sizes):
            # Make sure the key is legit -- that it does not overwrite
            # anything.
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

    There are several "reserved" names for expressions.

      - ``inpt``: observations of a supervised or unsupervised model,
      - ``target``: desired outputs of a supervised model,
      - ``loss``: quantity to be optimized for fitting the parameters;
        might not refer to the criterion of interest, but instead to a
        regularzied objective.
      - ``true_loss``: Quantity of interest for the user, e.g. the loss without
        regularization or the empirical risk.

    Overriding these names is possible in general, but is part of the interface
    and will lead to unexpected behaviour with functionality building upon
    this.


    Attributes
    ----------

    pars : ParameterSet object
        Holding the adaptable parameters of the object.

    exprs : dictionary
        Containig the expressions. Out of convenience, the external variables
        are held in here as well.

    updates : dict
        Containing update variables, e.g. due to the use of ``theano.scan``.
    """

    def __init__(self):
        self.updates = collections.defaultdict(dict)
        self._init_pars()
        self._init_exprs()

        # This is a dictionary which is supposed to hold substitions of
        # variables from .exprs for the use with the GPU.
        self.gpu_variable_subs = {}

    def _init_pars(self):
        pass

    def _init_exprs(self):
        pass

    def _unify_variables(self, variables):
        """Given a list of variables where each identifier given as a string
        is repaced by the corresponding variable from the .exprs
        dictionary."""
        def lookup(varname):
            res = getattr(self.parameters, varname, None)
            if res is None:
                res = self.exprs[i]
            return res
        variables = [lookup(i) if isinstance(i, str) else i
                     for i in variables]
        return variables

    def _unify_exprs(self, exprs):
        """Expressions can be identified either as a reference to the
        expression object or by its name in the .exprs dictionary.  In either
        case, it can also be a mixed list of both when passed to
        Model.function.

        This function unifies all possible arguments in that way and returns a
        list of proper expressions."""
        if isinstance(exprs, (str, unicode)):
            # We are only being given a single string expression.
            exprs = self.exprs[exprs]
        elif isinstance(exprs, list):
            # We have several, either string or variable, thus make it a list
            # and substitute the strings.
            exprs = list(exprs)
            exprs = [self.exprs[i] if isinstance(i, str) else i for i in exprs]
        else:
            exprs = exprs

        return exprs

    def function(self, variables, exprs, mode=None, explicit_pars=False,
                 givens=None,
                 on_unused_input='raise', numpy_result=False):
        """Return a compiled function for the given `exprs` given `variables`.


        Parameters
        ----------

        variables : list of strings
            Each string refers to an item in ``.exprs`` and is considered an
            input to the function.

        exprs : (List of) Theano expression or string
            Expressions for which to create the function. If a single
            expression is given, the function will return a single value; if a
            list is given, the result will be a tuple containing one element
            for each.  An expression can either be a Theano expression or a
            string. In the latter case, the corresponding expression will be
            retrieved from ``.exprs``.

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

        numpy_result : boolean, optional, default: False
            If set to True, a numpy array is always returned, even if the
            computation is done on the GPU and a gnumpy array was more natural.
        """
        variables = self._unify_variables(variables)
        exprs = self._unify_exprs(exprs)

        if GPU:
            back_out = False
            if isinstance(exprs, list):
                if not all(tell_deterministic(i) for i in exprs):
                    back_out = True
            else:
                if not tell_deterministic(exprs):
                    back_out = True
            if back_out:
                raise NotImplementedError(
                    'cannot use random variables in Breze for GPU due to Theano '
                    'issue #1467')

        # We need to clone instead of using the givens parameter of
        # theano.function, because otherwise we might get an theano error
        # with conflicting replacements. (See theano/compile/pfunc.py:162,
        # rebuild_collect_shared.)
        if givens is not None:
            if isinstance(exprs, list):
                exprs = [theano.clone(e, givens) for e in exprs]
            else:
                exprs = theano.clone(exprs, givens)
        else:
            givens = {}

        # Build update dictionary.
        updates = collections.defaultdict(lambda: {})
        if isinstance(exprs, (list, tuple)):
            flat_exprs = flatten(exprs)
            for expr in flat_exprs:
                # TODO: last takes all, maybe should throw an error.
                updates.update(self.updates[expr])
        else:
            updates.update(self.updates[exprs])

        if GPU:
            outputs = not numpy_result
            variables, exprs = self.var_exp_for_gpu(
                variables, exprs, outputs=outputs)

        variables = [self.parameters.flat] + variables

        f = theano_function_with_nested_exprs(
            variables, exprs, givens=givens, mode=mode,
            on_unused_input=on_unused_input, updates=updates)

        if GPU:
            f = gnumpy_func_wrap(f)

        if not explicit_pars:
            def f_implicit_pars(*args, **kwargs):
                return f(self.parameters.data, *args, **kwargs)
            f_implicit_pars.theano_func = f.theano_func
            f_implicit_pars.breze_func = True
            return f_implicit_pars

        else:
            f.breze_func = True
            return f

    def var_exp_for_gpu(self, variables, exprs, outputs=True):
        """Given variables and theano expressions built from these variables,
        return variables and expressions of the same form that are tailored
        towards GPU usage."""

        # Here is the outline of this function.
        #
        # (1) For each CPU tensor from theano.tensor create a corresponding GPU
        #     tensor from theano.sandbox.cuda,
        # (2) replace these in all expressions,
        # (3) replace the output expressions with GPU expressions so no
        #     auto-conversion to numpy is done.
        #
        # Since variables and expressions might be nested, we need to flatten
        # them first and unflatten the results.

        # Stage (1)
        variables_flat = flatten(variables)
        gpu_var_flat = []
        for var in variables_flat:
            if var in self.gpu_variable_subs:
                gpu_var = self.gpu_variable_subs[var]
            else:
                gpu_var = cpu_tensor_to_gpu(var)
                gpu_var.name = var.name + '-for-gpu'
                self.gpu_variable_subs[var] = gpu_var
            gpu_var_flat.append(gpu_var)
        gpu_variables = unflatten(variables, gpu_var_flat)

        # Loop for stage (2) and (3):
        exprs_flat = flatten(exprs)
        gpu_exprs_flat = []
        for expr in exprs_flat:
            # (2)
            for v, gv in zip(variables_flat, gpu_var_flat):
                expr = theano.clone(expr, {v: gv})
            # (3)
            if outputs:
                expr = cpu_expr_to_gpu(expr)
            gpu_exprs_flat.append(expr)

        gpu_exprs = unflatten(exprs, gpu_exprs_flat)

        return gpu_variables, gpu_exprs

    def __getstate__(self):
        state = self.__dict__.copy()
        to_delete = [k for k in state if getattr(state[k], 'breze_func', False)]
        for key in to_delete:
            del state[key]

        return state


class PrintEverythingMode(theano.Mode):
    def __init__(self):
        def print_eval(i, node, fn):
            print '<' * 50
            print i, node, [input[0] for input in fn.inputs],
            fn()
            print [output[0] for output in fn.outputs]
            print '>' * 50
        wrap_linker = theano.gof.WrapLinkerMany(
            [theano.gof.OpWiseCLinker()], [print_eval])
        super(PrintEverythingMode, self).__init__(
            wrap_linker, optimizer='fast_compile')


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
        wrap_linker = theano.gof.WrapLinkerMany(
            [theano.gof.OpWiseCLinker()], [print_eval])
        super(WarnNaNMode, self).__init__(
            wrap_linker, optimizer='fast_compile')
