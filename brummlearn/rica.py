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
from brummlearn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin, 
    ReconstructBrezeWrapperMixin)


class Rica(_Rica, UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin):

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

