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
        """Return the prediction of the network given input sequences.

        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        :param std: If True, returns the stanard deviation of the prediction as
            well.
        :returns: A (t, n, l) array where _t_ and _n_ are defined as in _X_,
            but _l_ is the dimensionality of the output sequences at a single
            time step.
        """
        if self.f_predict is None:
            self.f_predict, self.f_predict_std = self._make_predict_functions(
                self.stored_X, self.stored_Z)

        if std:
            return self.f_predict_std(X - self.mean_x) + self.mean_z
        else:
            return self.f_predict(X - self.mean_x) + self.mean_z
