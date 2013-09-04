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
import theano
import theano.tensor as T

from breze.arch.model.feature import Rica as _Rica
from breze.learn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)


class Rica(_Rica, UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin):

    def __init__(self, n_inpt, n_feature, hidden_transfer='identity',
                 feature_transfer='softabs', out_transfer='identity',
                 loss='squared', c_ica=0.5, optimizer='lbfgs',
                 max_iter=1000, verbose=False):
        """Create a Rica object.

        :param n_inpt: Input dimensionality of the data.
        :param n_feature: Dimensionality of the hidden feature dimension.
        :param hidden_transfer: Transfer function to use for the hidden units.
            Can be a string referring any function found in
            ``breze.component.transfer`` or a function that given an (n, d)
            array returns an (n, d) array as theano expressions.
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
        super(Rica, self).__init__(
            n_inpt, n_feature, hidden_transfer, feature_transfer,
            out_transfer, loss, c_ica)
        self.optimizer = optimizer
        self.f_transform = None
        self.f_reconstruct = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)
        self.max_iter = max_iter
        self.verbose = verbose

    def iter_fit(self, X):
        w = self.parameters['in_to_hidden']
        f_weights_normed = self.function([], 'in_to_hidden_normed')
        for info in super(Rica, self).iter_fit(X):
            w[...] = f_weights_normed()
            yield info
