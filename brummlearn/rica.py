# -*- coding: utf-8 -*-

"""ICA with reconstruction cost.

As introduced in

    ICA with Reconstruction Cost for Efficient Overcomplete Feature Learning.
    Quoc V. Le, Alex Karpenko, Jiquan Ngiam and Andrew Y. Ng.
    In NIPS*2011. 
"""

import itertools 

import climin
import numpy as np
import theano.tensor as T

from breze.model.feature import Rica as _Rica


class Rica(_Rica):

    def __init__(self, n_inpt, n_feature, feature_transfer='softabs',
        out_transfer='identity', loss='squared', c_ica=0.5,
        max_iter=1000, verbose=False):
        """Create a Rica object.

        :param n_inpt: Input dimensionality of the data.
        :param n_feature: Dimensionality of the hidden feature dimension.
        :param feature_transfer: Transfer function to use for the features.
            Can be a string referring any function found in
            ``breze.component.transfer`` or a function that given an (n, d)
            array returns an (n, d) array as theano expressions.
        :param out_transfer: Output transfer function of the linear auto encoder
            for calculation of the reconstruction cost.
        :param loss: Loss which is going to be optimized. This can either be a
            string and reference a loss function found in
            ``breze.component.distance`` or a function which takes two theano
            tensors (one being the output of the network, the other some target)
            and returns a theano scalar.
        :param c_ica: Weight of the ICA cost, cost of linear reconstruction is
            1.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(Rica, self).__init__(n_inpt, n_feature, feature_transfer,
            out_transfer, loss, c_ica)
        self.f_transform = None
        self.f_reconstruct = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape)
        self.max_iter = max_iter
        self.verbose = verbose

    def _d_loss(self):
        """Return a theano expression for the gradient of the loss wrt the
        flat parameters of the model."""
        return T.grad(self.exprs['loss'], self.parameters.flat)

    def _make_loss_functions(self):
        """Return pair (f_loss, f_d_loss) of functions.
        
         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        d_loss = self._d_loss()

        f_loss = self.function(['inpt'], 'loss', explicit_pars=True)
        f_d_loss = self.function(['inpt'], d_loss, explicit_pars=True)
        return f_loss, f_d_loss

    def _make_transform_function(self):
        """Return a callable f which does the feature transform of this model.
        """
        f_transform = self.function(['inpt'], 'feature')
        return f_transform

    def _make_reconstruct_function(self):
        """Return a callable f which does the reconstruction of this model.
        """
        f_reconstruct = self.function(['inpt'], 'output')
        return f_transform

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
