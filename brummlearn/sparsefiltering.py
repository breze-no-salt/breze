# -*- coding: utf-8 -*-

"""Sparse Filtering.

As introduced in

    Sparse Filtering
    Jiquan Ngiam, Pangwei Koh, Zhenghao Chen, Sonia Bhaskar and Andrew Y. Ng.
    In NIPS*2011.

"""

import numpy as np
import theano

from breze.model.feature import SparseFiltering as _SparseFiltering
from brummlearn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin)


class SparseFiltering(
    _SparseFiltering, UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin):

    def __init__(self, n_inpt, n_feature, feature_transfer='softabs',
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create a SparseFiltering object.

        :param n_inpt: Input dimensionality of the data.
        :param n_feature: Dimensionality of the hidden feature dimension.
        :param feature_transfer: Transfer function to use. Can be a string
            referring any function found in ``breze.component.transfer`` or
            a function that given an (n, d) array returns an (n, d) array as
            theano expressions.

            Should be symmetric.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(SparseFiltering, self).__init__(
            n_inpt, n_feature, feature_transfer)
        self.f_transform = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose
