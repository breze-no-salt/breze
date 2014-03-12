# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T

from breze.arch.model import gaussianprocess
from breze.arch.util import ParameterSet, Model
from breze.learn.base import SupervisedBrezeWrapperBase
from breze.learn.sampling import slice_
from breze.learn.utils import theano_floatx


class GaussianProcess(Model, SupervisedBrezeWrapperBase):
    """GaussianProcess class.

    Parameters
    ----------

    n_inpt : scalar
        Input dimensionality of a single input.

    kernel : string or function, optional
        Kernel to use. Can be a string which is then looked up in
        ``breze.arch.component.kernel``. Can also be a function that has the
        same interface.

    optimizer : string, or tuple of the form (string, dict), optional
        Arguments for ``climin.util.optimizer`` to construct an optimizer. See
        the docs for the exact behaviour.

    max_iter : int, optional
        Maximum number of optimization iterations to perform. Only respected
        if ``.fit()`` is used, not in the case of ``.iter_fit()``.

    verbose : boolean, optional
        Flag indicating whether to print out information during fitting.


    Examples
    --------

    See ``notebooks/Gaussian process on toy data.ipynb`` for an example.


    Notes
    -----
    The implementation is based on Kevin Murphy's book [MLPP]_.


    References
    ----------
    .. [MLPP] `Kevin Murphy. Machine Learning---A Propabilistic Perspective`.
        (2012)
        http://www.cs.ubc.ca/~murphyk/MLbook/index.html
    """

    def __init__(self, n_inpt, kernel='linear', optimizer='rprop',
                 max_iter=1000, verbose=False):
        self.n_inpt = n_inpt
        self.kernel = kernel

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose
        self._gram_matrix = None
        self.f_gram_matrix = None

        super(GaussianProcess, self).__init__()

    def _init_pars(self):
        spec = gaussianprocess.parameters(self.n_inpt)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:][...] = 0

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
            'test_inpt': T.matrix('inpt'),
            'target': T.matrix('target'),
        }
        P = self.parameters

        self.exprs.update(gaussianprocess.exprs(
            self.exprs['inpt'], self.exprs['test_inpt'], self.exprs['target'],
            P.length_scales, P.noise, P.amplitude, self.kernel))

    def _make_predict_functions(self, stored_inpt, stored_target):
        """Return a function to predict targets from input sequences."""
        if self.f_gram_matrix is None:
            self.f_gram_matrix = self.function(['inpt'], 'gram_matrix')

        if self._gram_matrix is None:
            self._gram_matrix = self.f_gram_matrix(stored_inpt)

        givens = {
            self.exprs['gram_matrix']: theano.shared(
                self._gram_matrix, name='gram_matrix-sub'),
            self.exprs['target']: theano.shared(
                stored_target, name='stored-target'),
            self.exprs['inpt']: theano.shared(
                stored_inpt, name='stored-inpt'),
        }

        # We ignore warnings for unused inputs in both cases. Why?
        # model.function will add the parameters vector to this function; but
        # since we substitute the kernel matrix expression (which is the only
        # way the parameters play a role) with a precomputed one, they will not
        # be part of the computational graph anymore.
        f_predict = self.function(['test_inpt'], 'output', givens=givens,
                                  on_unused_input='ignore')
        f_predict_var = self.function(
            ['test_inpt'], ['output', 'output_var'], givens=givens,
            on_unused_input='ignore')

        return f_predict, f_predict_var

    def store_dataset(self, X, Z):
        """Store the training set in the object.

        Gaussian processes are non-parametric models and thus rely on a set of
        training points. This set will be used for predictions and sampling from
        the posterior.

        Parameters
        ----------

        X : array_like
            Array of shape ``(n_sample, n_inpt)`` containing the training data.
        Z : array_like
            Array of shape ``(n_sample, 1)`` containing the target values.
        """

        self._gram_matrix = None
        self.mean_x = X.mean(axis=0)
        self.mean_z = Z.mean(axis=0)
        self.std_x = X.std(axis=0)
        self.std_z = Z.std(axis=0)
        self.stored_X = (X - self.mean_x) / self.std_x
        self.stored_Z = (Z - self.mean_z) / self.std_z

    def iter_fit(self, X, Z, mode=None):
        self.store_dataset(X, Z)

        if 'diff' in self.exprs:
            f_diff = self.function(['inpt'], 'diff')
            diff = f_diff(X)
            givens = {self.exprs['diff']: diff}
        else:
            givens = {}
        f_loss, f_d_loss = self._make_loss_functions(
            givens=givens, mode=mode, on_unused_input='warn')

        args = self._make_args(self.stored_X, self.stored_Z)
        opt = self._make_optimizer(f_loss, f_d_loss, args)

        for i, info in enumerate(opt):
            yield info

    def predict(self, X, var=False, max_rows=1000):
        """Predict the target values.

        Parameters
        ----------

        X : array_like
            Array of shape (n_samples, n_features).

        var : boolean
            Flag indicating whether the variance of the predictions should be
            returned as well. In this case, the complexity rises from O(n) to
            O(n^2).

        max_rows : integer
            Maximum number of predictions to do in one step; a lower number
            might help performance.

        Returns
        -------
        mean : array_like
            Array of the form (n_samples, 1) containing the mean of the
            predictions.
        variance : array_like
            Only if ``var == True``. Array of shape (n_samples, 1) containing
            the variance of the predictions.
        """
        if self.f_predict is None or self.f_predict_var is None:
            self.f_predict, self.f_predict_var = self._make_predict_functions(
                self.stored_X, self.stored_Z)

        n_steps, rest = divmod(X.shape[0], max_rows)
        if rest != 0:
            n_steps += 1
        steps = [(i * max_rows, (i + 1) * max_rows) for i in range(n_steps)]

        X = (X - self.mean_x) / self.std_x

        if var:
            Y, = theano_floatx(np.empty((X.shape[0], 1)))
            Y_var, = theano_floatx(np.empty((X.shape[0], 1)))

            for start, stop in steps:
                this_x = X[start:stop]
                m, s = self.f_predict_var(this_x)
                Y[start:stop] = m
                Y_var[start:stop] = s
            Y = (Y * self.std_z) + self.mean_z
            Y_var = Y_var * self.std_z
            return Y, Y_var
        else:
            Y, = theano_floatx(np.empty((X.shape[0], 1)))
            for start, stop in steps:
                this_x = X[start:stop]
                Y[start:stop] = self.f_predict(this_x)
            Y = (Y * self.std_z) + self.mean_z
            return Y

    def sample_parameters(self):
        """Use slice sampling to sample a hyper parameters from the posterior
        given the observations.

        One step of slice sampling is performed with the current parameters as
        a starting point. The current parameters are overwritten by the sample.

        For sampling, the current stored data set is used.
        """
        if getattr(self, 'f_nll_expl', None) is None:
            self.f_nll_expl = self.function(['inpt', 'target'], 'nll', explicit_pars=True)

        f_ll = lambda pars: -self.f_nll_expl(pars, self.stored_X, self.stored_Z)

        self.parameters.data[:] = slice_.sample(f_ll, self.parameters.data, window_inc=1.)
        self._gram_matrix = None
        self.f_predict = None
        self.f_predict_var = None
