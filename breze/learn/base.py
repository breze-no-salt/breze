# -*- coding: utf-8 -*-

"""Module that contains base functionality for implementations of
learning algorithms."""


import itertools
import warnings
import signal

import climin
import climin.util
import climin.mathadapt as ma
import numpy as np
import theano
import theano.tensor as T

#from climin.util import iter_minibatches2 as iter_minibatches
from climin.util import iter_minibatches


GPU = theano.config.device.startswith('gpu')
if GPU:
    import gnumpy as gp


from breze.arch.util import Model


def cast_array_to_local_type(arr):
    """Given an array (HDF5, numpy, gnumpy) return an array that matches the
    current theano configuration.

    That is, if the current device is GPU, make it a gnumpy.garry. If the
    current theano.config.floatX does not match the dtype of arr, return an
    array that does."""
    res = arr
    if GPU and not isinstance(arr, gp.garray):
        warnings.warn('Implicilty converting numpy.ndarray to gnumpy.garray')
        res = gp.as_garray(res)
    elif isinstance(arr, np.ndarray) and arr.dtype != theano.config.floatX:
        res = arr.astype(theano.config.floatX)
    return res

theanox = cast_array_to_local_type


def assert_ndarray(arr):
    """If ``arr`` is a ``gnumpy.garray``, convert it to a ``numpy.ndarray``.
    Otherwise pass silently."""
    if GPU and isinstance(arr, gp.garray):
        arr = arr.as_numpy_array()
    return arr


def make_clipper(threshold):
    def clip_if_long(func):
        def inner(*args, **kwargs):
            res = func(*args, **kwargs)
            length = np.sqrt((res ** 2).sum())
            if  length > threshold:
                res /= length
                res *= threshold
            return res
        return inner
    return clip_if_long


class BrezeWrapperBase(object):
    """Class that helps with wrapping Breze models."""

    mode = None

    # Not all subclasses need to implement this. For this case, we supply a
    # default.
    imp_weight = False

    def _d_loss(self):
        """Return a theano expression for the gradient of the loss wrt the
        flat parameters of the model."""
        return T.grad(self.exprs['loss'], self.parameters.flat)

    def _make_optimizer(self, f, fprime, args, wrt=None, f_Hp=None, info=None):
        if isinstance(self.optimizer, (str, unicode)):
            ident = self.optimizer
            kwargs = {}
        else:
            ident, kwargs = self.optimizer

        # If we do not make this copy, we will add functions and a generator
        # (the args) to an instance variable. This will result in
        # unpicklability of the object.
        kwargs = kwargs.copy()

        kwargs['f'] = f
        kwargs['fprime'] = fprime

        if wrt is None:
            wrt = self.parameters.data

        if f_Hp is not None:
            kwargs['f_Hp'] = f_Hp

        kwargs['args'] = args
        opt = climin.util.optimizer(ident, wrt, **kwargs)
        if info is not None:
            opt.set_from_info(info)
        return opt

    def powerfit(self, fit_data, eval_data, stop, report, eval_train_loss=True):
        """Iteratively fit the model.

        This is a convenience function which combines iteratively fitting a
        model with stopping criterions and keeping track of the best parameters
        found so far.

        An iterator of dictionaries is returned; values are only yielded in the
        case that the call `report(info)` returns True. The iterator
        stops as soon as the call `stop(info)` returns True.

        Each dictionary yielded is directly obtained from the optimizer used to
        optimize the loss. It is augmented with the keys `loss`, `best_pars`
        and `best_loss`. The best loss is obtained by evaluating the loss of
        the model (given by model.exprs['loss']) on `eval_data`, while training
        is done on `fit_data`.

        This method respects a ``true_loss`` entry in the ``exprs``
        dictionary: if it is present, it will be used for reporting the loss
        and for comparing models throughout optimization instead of ``loss``,
        which will be used for the optimization itself. This makes it possible
        to add regularization terms to the loss and use other losses (such as
        the zero-one loss) for comparison of parameters.

        :param fit_data: A tuple containing arrays representing the data the
            model should be fitted on.
        :param eval_data: A tuple containing arrays representing the data the
            model should be evaluated on, and which gives which model is
            "best".
        :param stop: A function receiving an info dictionary which returns True
            if the iterator should stop.
        :param report: A function receiving an info dictionary which should
            return True if the iterator should yield a value.
        :returns: An iterator over info dictionaries.
        """
        # TODO document eval train loss
        self.CTRL_C_FLAG = False
        signal.signal(signal.SIGINT, self._ctrl_c_handler)

        best_pars = None
        best_loss = float('inf')

        for info in self.iter_fit(*fit_data):
            if report(info):
                if 'loss' not in info:
                    # Not all optimizers, e.g. ilne and gd, do actually
                    # calculate the loss.
                    if eval_train_loss:
                        info['loss'] = ma.scalar(self.score(*fit_data))
                    else:
                        info['loss'] = 0.
                info['val_loss'] = ma.scalar(self.score(*eval_data))

                if info['val_loss'] < best_loss:
                    best_loss = info['val_loss']
                    best_pars = self.parameters.data.copy()

                info['best_loss'] = best_loss
                info['best_pars'] = best_pars

                yield info

                if stop(info) or self.CTRL_C_FLAG:
                    break

    def _ctrl_c_handler(self, signal, frame):
        self.CTRL_C_FLAG = True


class SupervisedModel(Model, BrezeWrapperBase):

    data_arguments = 'inpt', 'target'
    sample_dim = 0, 0

    # Not all subclasses need to implement this. For this case, we supply a
    # default.
    imp_weight = False

    f_score = None
    f_predict = None
    _f_loss = None
    _f_dloss = None

    gradient_clip_threshold = None

    @property
    def inpt(self):
        return self.exprs['inpt']

    @property
    def output(self):
        return self.exprs['output']

    @property
    def target(self):
        return self.exprs['target']

    @property
    def loss(self):
        return self.exprs['loss']

    def __init__(self, inpt, target, output, loss, parameters):
        self.parameters = parameters
        self.parameters.alloc()

        self.exprs = {
            'inpt': inpt,
            'target': target,
            'output': output,
            'loss': loss,
        }

        super(SupervisedModel, self).__init__()

    def _make_loss_functions(self, mode=None, givens=None,
                             on_unused_input='raise', imp_weight=False):
        """Return pair (f_loss, f_d_loss) of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        d_loss = self._d_loss()
        givens = {} if givens is None else givens
        inpts = ['inpt', 'target']
        if imp_weight:
            inpts += ['imp_weight']
        f_loss = self.function(inpts, 'loss', explicit_pars=True,
                               mode=mode, givens=givens,
                               on_unused_input=on_unused_input)
        f_d_loss = self.function(
            inpts, d_loss, explicit_pars=True, mode=mode,
            givens=givens, on_unused_input=on_unused_input)

        if self.gradient_clip_threshold is not None:
            clipper = make_clipper(self.gradient_clip_threshold)
            f_d_loss = clipper(f_d_loss)

        return f_loss, f_d_loss

    def _make_args(self, X, Z, imp_weight=None, n_cycles=False):
        batch_size = getattr(self, 'batch_size', None)
        if batch_size is None:
            X, Z = cast_array_to_local_type(X), cast_array_to_local_type(Z)
            times = n_cycles if n_cycles else None
            if imp_weight is not None:
                imp_weight = cast_array_to_local_type(imp_weight)
                data = itertools.repeat([X, Z, imp_weight], times=times)
            else:
                data = itertools.repeat([X, Z], times=times)
        elif batch_size < 1:
            raise ValueError('need strictly positive batch size')
        else:
            if imp_weight is not None:
                data = iter_minibatches([X, Z, imp_weight], self.batch_size,
                                        list(self.sample_dim) + [self.sample_dim[0]], n_cycles=n_cycles)
                data = ((cast_array_to_local_type(x),
                         cast_array_to_local_type(z),
                         cast_array_to_local_type(w)) for x, z, w in data)
            else:
                data = iter_minibatches([X, Z], self.batch_size,
                                        self.sample_dim, n_cycles=n_cycles)

                data = ((cast_array_to_local_type(x),
                         cast_array_to_local_type(z)) for x, z in data)

        args = ((i, {}) for i in data)
        return args

    def iter_fit(self, X, Z, imp_weight=None, info_opt=None):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the
        returned iterator. The model is in a valid state after each iteration,
        so that the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.

        :param X: Array representing the inputs.
        :param Z: Array representing the outputs.
        """
        if imp_weight is None and self.imp_weight:
            raise ValueError('need to provide ``imp_weight``.')
        if imp_weight is not None and not self.imp_weight:
            raise ValueError('do not need ``imp_weight``.')

        if self._f_loss is None or self._f_dloss is None:
            self._f_loss, self._f_dloss = self._make_loss_functions(
                imp_weight=(imp_weight is not None))

        args = self._make_args(X, Z, imp_weight)
        opt = self._make_optimizer(self._f_loss, self._f_dloss, args, info=info_opt)

        for i, info in enumerate(opt):
            yield info

    def fit(self, X, Z, imp_weight=None):
        """Fit the parameters of the model to the given data with the
        given error function.

        :param X: Array representing the inputs.
        :param Z: Array representing the outputs.
        """
        itr = self.iter_fit(X, Z, imp_weight)
        if self.verbose:
            print 'Optimizing for %i iterations.' % self.max_iter
        for i, info in enumerate(itr):
            if self.verbose:
                if 'loss' in info:
                    print '%i/%i %g' % (info['n_iter'], self.max_iter,
                                        info['loss'])
                else:
                    print '%i/%i' % (info['n_iter'], self.max_iter)
            if i + 1 >= self.max_iter:
                break

    def _make_predict_functions(self):
        """Return a function to predict targets from input sequences."""
        return self.function(['inpt'], 'output')

    def predict(self, X):
        """Return the prediction of the model given the input.

        Parameters
        ----------

        X : array_like
            Input to the model.

        Returns
        -------

        Y : array_like
        """
        X = cast_array_to_local_type(X)
        if self.f_predict is None:
            self.f_predict = self._make_predict_functions()
        Y = self.f_predict(X)

        return Y

    def _make_score_function(self, imp_weight=False):
        """Return a function to predict targets from input sequences."""
        key = 'true_loss' if 'true_loss' in self.exprs else 'loss'
        inpts = ['inpt', 'target']
        if imp_weight:
            inpts += ['imp_weight']
        return self.function(inpts, key)

    def score(self, X, Z, imp_weight=None):
        """Return the score of the model given the input and targets.

        Parameters
        ----------

        X : array_like
            Input to the model.

        Z : array_like
            Target for the inputs.

        Returns
        -------

        l : scalar
            Score of the model.
        """
        if self.f_score is None:
            self.f_score = self._make_score_function(
                imp_weight=(imp_weight is not None))

        score = 0
        sample_count = 0
        for arg in self._make_args(X, Z, imp_weight, n_cycles=1):
            samples_in_batch = int(arg[0][0].shape[self.sample_dim[0]])
            score += self.f_score(*arg[0]) * samples_in_batch
            sample_count += samples_in_batch
        return score / sample_count


class UnsupervisedModel(Model, BrezeWrapperBase):

    data_arguments = 'inpt',
    sample_dim = 0,
    f_score = None
    _f_loss = None
    _f_dloss = None

    gradient_clip_threshold = None

    @property
    def inpt(self):
        return self.exprs['inpt']

    @property
    def output(self):
        return self.exprs['output']

    @property
    def loss(self):
        return self.exprs['loss']

    def __init__(self, inpt, output, loss, parameters, imp_weight=None):
        self.parameters = parameters
        self.parameters.alloc()

        self.exprs = {
            'inpt': inpt,
            'output': output,
            'loss': loss,
            'imp_weight': imp_weight
        }

        super(UnsupervisedModel, self).__init__()

    def _make_loss_functions(self, mode=None, givens=None,
                             on_unused_input='raise', imp_weight=False):
        """Return pair (f_loss, f_d_loss) of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        d_loss = self._d_loss()
        givens = {} if givens is None else givens

        args = ['inpt'] if not imp_weight else ['inpt', 'imp_weight']

        f_loss = self.function(args, 'loss', explicit_pars=True, mode=mode,
                               givens=givens, on_unused_input=on_unused_input)
        f_d_loss = self.function(
            args, d_loss, explicit_pars=True, givens=givens, mode=mode,
            on_unused_input=on_unused_input)

        if self.gradient_clip_threshold is not None:
            clipper = make_clipper(self.gradient_clip_threshold)
            f_d_loss = clipper(f_d_loss)

        return f_loss, f_d_loss

    def iter_fit(self, X, W=None, info_opt=None):
        """Iteratively fit the parameters of the model to the given data.

        Each iteration of the learning algorithm is an iteration of the
        returned iterator. The model is in a valid state after each iteration,
        so that the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.

        :param X: Array representing the samples.
        """
        use_imp_weight = W is not None

        if self._f_loss is None or self._f_dloss is None:
            self._f_loss, self._f_dloss = self._make_loss_functions(
                imp_weight=use_imp_weight)

        if W is None and self.imp_weight:
            raise ValueError('need to provide ``W``.')
        if W is not None and not self.imp_weight:
            raise ValueError('do not need ``W``.')

        arg_args = [X, W] if use_imp_weight else [X]
        args = self._make_args(*arg_args)
        opt = self._make_optimizer(self._f_loss, self._f_dloss, args, info=info_opt)

        for i, info in enumerate(opt):
            yield info

    def fit(self, X, W=None):
        """Fit the parameters of the model.

        :param X: Array representing the samples.
        """
        iter_fit_args = [X] if W is None else [X, W]
        itr = self.iter_fit(*iter_fit_args)
        if self.verbose:
            print 'Optimizing for %i iterations.' % self.max_iter
        for i, info in enumerate(itr):
            if self.verbose:
                if 'loss' in info:
                    print '%i/%i %g' % (info['n_iter'], self.max_iter,
                                        info['loss'])
                else:
                    print '%i/%i' % (info['n_iter'], self.max_iter)
            if i + 1 >= self.max_iter:
                break

    def _make_args(self, X, W=None, n_cycles=False):
        batch_size = getattr(self, 'batch_size', None)
        use_imp_weight = W is not None
        if self.use_imp_weight != use_imp_weight:
            raise ValueError('need to give ``W`` in accordinace to '
                             '``self.imp_weight``')
        item = [X, W] if use_imp_weight else [X]

        # If importance weights are used, we need to append the sample
        # dimensionality of it, which will be the same as for that data.
        sample_dim = list(self.sample_dim)
        if use_imp_weight:
            sample_dim.append(sample_dim[0])

        if batch_size is None:
            times = n_cycles if n_cycles else None
            data = itertools.repeat(item, times=times)
        elif batch_size < 1:
            raise ValueError('need strictly positive batch size')
        else:
            data = iter_minibatches(item, self.batch_size, sample_dim, n_cycles=n_cycles)
        if use_imp_weight:
            data = ((cast_array_to_local_type(x), cast_array_to_local_type(w))
                    for x, w in data)
        else:
            data = ((cast_array_to_local_type(x),)
                    for x, in data)
        args = ((i, {}) for i in data)
        return args

    def _make_score_function(self):
        """Return a function to predict targets from input sequences."""
        key = 'true_loss' if 'true_loss' in self.exprs else 'loss'
        args = ['inpt'] if not self.imp_weight else ['inpt', 'imp_weight']
        return self.function(args, key)

    def score(self, X, W=None):
        """Return the score of the model given the input and targets.

        Parameters
        ----------

        X : array_like
            Input to the model.

        Returns
        -------

        l : scalar
            Score of the model.
        """
        if self.f_score is None:
            self.f_score = self._make_score_function()

        score = 0
        sample_count = 0
        for arg in self._make_args(X, W, n_cycles=1):
            samples_in_batch = int(arg[0][0].shape[self.sample_dim[0]])
            score += self.f_score(*arg[0]) * samples_in_batch
            sample_count += samples_in_batch
        return score / sample_count


class TransformBrezeWrapperMixin(object):

    transform_expr_name = 'feature'
    f_transform = None

    def _make_transform_function(self):
        """Return a callable f which does the feature transform of this model.
        """
        f_transform = self.function(['inpt'], self.transform_expr_name)
        return f_transform

    def transform(self, X):
        """Return the feature representation of the model given X.

        Parameters
        ----------

        X : array_like
            Represents the inputs to be transformed.

        Returns
        -------

        Y : array_like
            Transformation of X under the model.
        """
        X = cast_array_to_local_type(X)
        if self.f_transform is None:
            self.f_transform = self._make_transform_function()
        Y = self.f_transform(X)

        return Y


class ReconstructBrezeWrapperMixin(object):

    def _make_reconstruct_function(self):
        """Return a callable f which does the reconstruction of this model.
        """
        f_reconstruct = self.function(['inpt'], 'output')
        return f_reconstruct

    def reconstruct(self, X):
        """Return the input reconstruction of the model given X.

        :param X: An array representing the inputs.
        :returns: An array representing the reconstructions of the input.
        """
        if self.f_reconstruct is None:
            self.f_reconstruct = self._make_reconstruct_function()
        return self.f_reconstruct(X)
