# -*- coding: utf-8 -*-


import itertools


import climin
import climin.util
import climin.gd

import numpy as np
import theano
import theano.tensor as T

from breze.model.gaussianprocess import GaussianProcess as GaussianProcess_

from brummlearn.base import SupervisedBrezeWrapperBase


class GaussianProcess(GaussianProcess_, SupervisedBrezeWrapperBase):

    def __init__(self, n_inpt, kernel='linear', noise=1e-6,
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
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        super(GaussianProcess, self).__init__(
            n_inpt, kernel=kernel, noise=noise)

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None
        self.f_predict_std = None

    def _make_predict_functions(self, stored_inpt, stored_target):
        """Return a function to predict targets from input sequences."""
        givens = {
            self.exprs['inpt']: theano.shared(stored_inpt),
            self.exprs['target']: theano.shared(stored_target)
        }

        f_predict = self.function(['test_inpt'], 'output', givens=givens)
        f_predict_std = self.function(
            ['test_inpt'], ['output', 'output_std'], givens=givens)

        return f_predict, f_predict_std

    def iter_fit(self, X, Z):
        self.mean_x = X.mean(axis=0)
        self.mean_z = Z.mean(axis=0)
        self.stored_X = X - self.mean_x
        self.stored_Z = Z - self.mean_z

        f_loss, f_d_loss = self._make_loss_functions()

        args = self._make_args(X, Z)
        opt = self._make_optimizer(f_loss, f_d_loss, args)

        for i, info in enumerate(opt):
            yield info

    def predict(self, X, std=False):
        """Return the prediction of the Gaussian process given input sequences.

        :param X: A (n, d) array where _n_ is the number of data samples and
            _d_ is the dimensionality of a data sample.
        :param std: If True, returns the stanard deviation of the prediction as
            well.
        :returns: A (n, 1) array where _n_ is the same as in _X_.
        """
        if self.f_predict is None:
            self.f_predict, self.f_predict_std = self._make_predict_functions(
                self.stored_X, self.stored_Z)

        if std:
            return self.f_predict_std(X - self.mean_x) + self.mean_z
        else:
            return self.f_predict(X - self.mean_x) + self.mean_z
