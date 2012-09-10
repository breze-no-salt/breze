# -*- coding: utf-8 -*-

"""Module that contains base functionality for implementations of
learning algorithms."""


import itertools

import climin
import climin.util
import theano.tensor as T

from brummlearn.data import iter_minibatches


class BrezeWrapperBase(object):
    """Class that helps with wrapping Breze models."""

    def _d_loss(self):
        """Return a theano expression for the gradient of the loss wrt the
        flat parameters of the model."""
        return T.grad(self.exprs['loss'], self.parameters.flat)

    def _make_optimizer(self, f, fprime, args):
        if isinstance(self.optimizer, (str, unicode)):
            ident = self.optimizer
            kwargs = {}
        else:
            ident, kwargs = self.optimizer
        kwargs['f'] = f
        kwargs['fprime'] = fprime

        kwargs['args'] = args
        return climin.util.optimizer(ident, self.parameters.data, **kwargs)


class SupervisedBrezeWrapperBase(BrezeWrapperBase):

    def _make_loss_functions(self, mode='fast_run'):
        """Return pair (f_loss, f_d_loss) of functions.
        
         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        d_loss = self._d_loss()

        f_loss = self.function(['inpt', 'target'], 'loss', explicit_pars=True,
                               mode=mode)
        f_d_loss = self.function(['inpt', 'target'], d_loss, explicit_pars=True,
                                 mode=mode)
        return f_loss, f_d_loss

    def _make_args(self, X, Z):
        if getattr(self, 'batch_size', None) is None:
            data = itertools.repeat([X, Z])
        else:
            data = iter_minibatches([X, Z], self.batch_size, (0, 0))
        args = ((i, {}) for i in data)
        return args

    def iter_fit(self, X, Z):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the returned
        iterator. The model is in a valid state after each iteration, so that
        the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.
        
        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        :param Z: A (t, n, l) array where _t_ and _n_ are defined as in _X_,
            but _l_ is the dimensionality of the output sequences at a single
            time step.
        """
        f_loss, f_d_loss = self._make_loss_functions()

        args = self._make_args(X, Z)
        opt = self._make_optimizer(f_loss, f_d_loss, args)

        for i, info in enumerate(opt):
            loss = info.get('loss', None)
            if loss is None:
                loss = f_loss(self.parameters.data, X, Z)
            info['loss'] = loss
            yield info

    def fit(self, X, Z):
        """Fit the parameters of the model to the given data with the
        given error function.

        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        :param Z: A (t, n, l) array where _t_ and _n_ are defined as in _X_,
            but _l_ is the dimensionality of the output sequences at a single
            time step.
        """
        itr = self.iter_fit(X, Z)
        for i, info in enumerate(itr):
            if i + 1 >= self.max_iter:
                break


    def _make_predict_functions(self):
        """Return a function to predict targets from input sequences."""
        return self.function(['inpt'], 'output')


    def predict(self, X):
        """Return the prediction of the network given input sequences.

        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        :returns: A (t, n, l) array where _t_ and _n_ are defined as in _X_,
            but _l_ is the dimensionality of the output sequences at a single
            time step.
        """
        if self.f_predict is None:
            self.f_predict = self._make_predict_functions()
        return self.f_predict(X)


class UnsupervisedBrezeWrapperBase(BrezeWrapperBase):

    def iter_fit(self, X):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the returned
        iterator. The model is in a valid state after each iteration, so that
        the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.
        
        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        """
        f_loss, f_d_loss = self._make_loss_functions()

        args = itertools.repeat(([X], {}))
        opt = climin.Lbfgs(self.parameters.data, f_loss, f_d_loss, args=args)

        for i, info in enumerate(opt):
            loss = info.get('loss', None)
            if loss is None:
                loss = f_loss(self.parameters.data, X)
            info['loss'] = loss
            yield info

    def fit(self, X):
        """Fit the parameters of the model to the given data with the
        given error function.

        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        """
        itr = self.iter_fit(X)
        for i, info in enumerate(itr):
            if i + 1 >= self.max_iter:
                break

    def _make_loss_functions(self):
        """Return pair (f_loss, f_d_loss) of functions.
        
         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        d_loss = self._d_loss()

        f_loss = self.function(['inpt'], 'loss', explicit_pars=True)
        f_d_loss = self.function(['inpt'], d_loss, explicit_pars=True)
        return f_loss, f_d_loss


class TransformBrezeWrapperMixin(object):

    def _make_transform_function(self):
        """Return a callable f which does the feature transform of this model.
        """
        f_transform = self.function(['inpt'], 'feature')
        return f_transform

    def transform(self, X):
        """Return the feature representation of the model given X.
        
        :param X: (n, d) array where n is the number of samples and d the input
            dimensionality.
        :returns:  (n, h) array where n is the number of samples and h the
            dimensionality of the feature space.
        """
        if self.f_transform is None:
            self.f_transform = self._make_transform_function()
        return self.f_transform(X)


class ReconstructBrezeWrapperMixin(object):

    def _make_reconstruct_function(self):
        """Return a callable f which does the reconstruction of this model.
        """
        f_reconstruct = self.function(['inpt'], 'output')
        return f_reconstruct

    def reconstruct(self, X):
        """Return the input reconstruction of the model given X.
        
        :param X: (n, d) array where n is the number of samples and d the input
            dimensionality.
        :returns:  (n, d) array where n is the number of samples and d the
            dimensionality of the input space.
        """
        if self.f_reconstruct is None:
            self.f_reconstruct = self._make_reconstruct_function()
        return self.f_reconstruct(X)
