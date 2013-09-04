# -*- coding: utf-8 -*-

"""Module that contains base functionality for implementations of
learning algorithms."""


import itertools
import warnings

import climin
import climin.util
import climin.mathadapt as ma
import numpy as np
import theano
import theano.tensor as T


from brummlearn.data import iter_minibatches

GPU = theano.config.device == 'gpu'
if GPU:
    import gnumpy as gp


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


def assert_ndarray(arr):
    """If ``arr`` is a ``gnumpy.garray``, convert it to a ``numpy.ndarray``.
    Otherwise pass silently."""
    if GPU and isinstance(arr, gp.garray):
        arr = arr.as_numpy_array()
    return arr


class BrezeWrapperBase(object):
    """Class that helps with wrapping Breze models."""

    mode = None

    def _d_loss(self):
        """Return a theano expression for the gradient of the loss wrt the
        flat parameters of the model."""
        return T.grad(self.exprs['loss'], self.parameters.flat)

    def _make_optimizer(self, f, fprime, args, wrt=None, f_Hp=None):
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
        return climin.util.optimizer(ident, wrt, **kwargs)

    def powerfit(self, fit_data, eval_data, stop, report):
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
        loss_key = 'true_loss' if 'true_loss' in self.exprs else 'loss'
        f_loss = self.function(self.data_arguments, loss_key)

        best_pars = None
        best_loss = float('inf')

        fit_data = [cast_array_to_local_type(i) for i in fit_data]
        eval_data = [cast_array_to_local_type(i) for i in eval_data]

        for info in self.iter_fit(*fit_data):
            if report(info):
                if 'loss' not in info:
                    # Not all optimizers, e.g. ilne and gd, do actually
                    # calculate the loss.
                    info['loss'] = ma.scalar(f_loss(*fit_data))
                info['val_loss'] = ma.scalar(f_loss(*eval_data))

                if info['val_loss'] < best_loss:
                    best_loss = info['val_loss']
                    best_pars = self.parameters.data.copy()

                info['best_loss'] = best_loss
                info['best_pars'] = best_pars

                yield info
            if stop(info):
                break


class SupervisedBrezeWrapperBase(BrezeWrapperBase):

    data_arguments = 'inpt', 'target'
    sample_dim = 0, 0

    def _make_loss_functions(self, mode=None, givens=None,
                             on_unused_input='raise'):
        """Return pair (f_loss, f_d_loss) of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        if mode is None:
            mode = self.mode

        d_loss = self._d_loss()
        givens = {} if givens is None else givens

        f_loss = self.function(['inpt', 'target'], 'loss', explicit_pars=True,
                               mode=mode, givens=givens,
                               on_unused_input=on_unused_input)
        f_d_loss = self.function(
            ['inpt', 'target'], d_loss, explicit_pars=True, mode=mode,
            givens=givens, on_unused_input=on_unused_input)

        return f_loss, f_d_loss

    def _make_args(self, X, Z):
        X, Z = cast_array_to_local_type(X), cast_array_to_local_type(Z)
        batch_size = getattr(self, 'batch_size', None)
        if batch_size is None:
            data = itertools.repeat([X, Z])
        elif batch_size < 1:
            raise ValueError('need strictly positive batch size')
        else:
            data = iter_minibatches([X, Z], self.batch_size, self.sample_dim)
        args = ((i, {}) for i in data)
        return args

    def iter_fit(self, X, Z):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the
        returned iterator. The model is in a valid state after each iteration,
        so that the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.

        :param X: Array representing the inputs.
        :param Z: Array representing the outputs.
        """
        f_loss, f_d_loss = self._make_loss_functions()

        args = self._make_args(X, Z)
        opt = self._make_optimizer(f_loss, f_d_loss, args)

        for i, info in enumerate(opt):
            yield info

    def fit(self, X, Z):
        """Fit the parameters of the model to the given data with the
        given error function.

        :param X: Array representing the inputs.
        :param Z: Array representing the outputs.
        """
        itr = self.iter_fit(X, Z)
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


class UnsupervisedBrezeWrapperBase(BrezeWrapperBase):

    data_arguments = 'inpt',
    sample_dim = 0,

    def iter_fit(self, X):
        """Iteratively fit the parameters of the model to the given data.

        Each iteration of the learning algorithm is an iteration of the
        returned iterator. The model is in a valid state after each iteration,
        so that the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.

        :param X: Array representing the samples.
        """
        f_loss, f_d_loss = self._make_loss_functions()

        args = self._make_args(X)
        opt = self._make_optimizer(f_loss, f_d_loss, args)

        for i, info in enumerate(opt):
            yield info

    def fit(self, X):
        """Fit the parameters of the model.

        :param X: Array representing the samples.
        """
        itr = self.iter_fit(X)
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

    def _make_args(self, X):
        batch_size = getattr(self, 'batch_size', None)
        if batch_size is None:
            data = itertools.repeat([X])
        elif batch_size < 1:
            raise ValueError('need strictly positive batch size')
        else:
            data = iter_minibatches([X], self.batch_size, self.sample_dim)
        args = ((i, {}) for i in data)
        return args

    def _make_loss_functions(self, mode=None, givens=None,
                             on_unused_input='raise'):
        """Return pair (f_loss, f_d_loss) of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        if mode is None:
            mode = self.mode

        d_loss = self._d_loss()
        givens = {} if givens is None else givens

        f_loss = self.function(['inpt'], 'loss', explicit_pars=True, mode=mode,
                               givens=givens, on_unused_input=on_unused_input)
        f_d_loss = self.function(
            ['inpt'], d_loss, explicit_pars=True, givens=givens, mode=mode,
            on_unused_input=on_unused_input)
        return f_loss, f_d_loss


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

        :param X: An array representing the inputs.
        :returns: An array representing the transformed inputs.
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
