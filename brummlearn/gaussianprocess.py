# -*- coding: utf-8 -*-


import numpy as np
import theano

from breze.model.gaussianprocess import GaussianProcess as GaussianProcess_

from brummlearn.base import SupervisedBrezeWrapperBase
from brummlearn.sampling import slice_


class GaussianProcess(GaussianProcess_, SupervisedBrezeWrapperBase):

    def __init__(self, n_inpt, kernel='linear', optimizer='rprop',
                 max_iter=1000, verbose=False):
        """Create a GaussianProcess object.

        :param n_inpt: Input dimensionality of a single input.
        :param kernel: String that identifies what kernel to use. Options are
            'linear', 'rbf' and 'matern52'.
        :param optimizer: Can be either a string or a pair. In any case,
            climin.util.optimizer is used to construct an optimizer. In the case
            of a string, the string is used as an identifier for the optimizer
            which is then instantiated with default arguments. If a pair,
            expected to be (`identifier`, `kwargs`) for more fine control of the
            optimizer.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(GaussianProcess, self).__init__(n_inpt, kernel=kernel)

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None
        self.f_predict_var = None
        self.f_gram_matrix = None

        self.parameters.data[:] = 0
        self._gram_matrix = None

    def _make_predict_functions(self, stored_inpt, stored_target):
        """Return a function to predict targets from input sequences."""
        if self.f_gram_matrix is None:
            self.f_gram_matrix = self.function(['inpt'], 'gram_matrix')

        if self._gram_matrix is None:
            self._gram_matrix = self.f_gram_matrix(stored_inpt)

        givens = {
            self.exprs['gram_matrix']: theano.shared(self._gram_matrix),
            self.exprs['target']: theano.shared(stored_target),
            self.exprs['inpt']: theano.shared(stored_inpt),
        }

        f_predict = self.function(['test_inpt'], 'output', givens=givens)
        f_predict_var = self.function(
            ['test_inpt'], ['output', 'output_var'], givens=givens)

        return f_predict, f_predict_var

    def store_dataset(self, X, Z):
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
        """Return the prediction of the Gaussian process given input sequences.

        :param X: A (n, d) array where _n_ is the number of data samples and
            _d_ is the dimensionality of a data sample.
        :param var: If True, returns the variance of the prediction as
            well.
        :param max_rows: Maximum number of predictions to do in one step; a
            lower number might help performance if the call stalls.
        :returns: A (n, 1) array where _n_ is the same as in _X_.
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
            Y = np.empty((X.shape[0], 1))
            Y_var = np.empty((X.shape[0], 1))
            for start, stop in steps:
                this_x = X[start:stop]
                m, s = self.f_predict_var(this_x)
                Y[start:stop] = m
                Y_var[start:stop] = s
            Y = (Y * self.std_z) + self.mean_z
            Y_var = Y_var * self.std_z

            return Y, Y_var
        else:
            Y = np.empty((X.shape[0], 1))
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

        :param X: A (n, d) array where _n_ is the number of data samples and
            _d_ is the dimensionality of a data sample containing the input
            data.
        :param Z: A (n, 1) array where _n_ is the number of data samples
            containing the output data.
        """
        if getattr(self, 'f_nll_expl', None) is None:
            self.f_nll_expl = self.function(['inpt', 'target'], 'nll', explicit_pars=True)

        f_ll = lambda pars: -self.f_nll_expl(pars, self.stored_X, self.stored_Z)

        self.parameters.data[:] = slice_.sample(f_ll, self.parameters.data, window_inc=1.)
        self._gram_matrix = None
        self.f_predict = None
        self.f_predict_var = None
