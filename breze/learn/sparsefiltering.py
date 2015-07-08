# -*- coding: utf-8 -*-

"""Sparse Filtering.

As introduced in

    Sparse Filtering
    Jiquan Ngiam, Pangwei Koh, Zhenghao Chen, Sonia Bhaskar and Andrew Y. Ng.
    In NIPS*2011.

"""

import numpy as np
import theano
import theano.tensor as T

from breze.arch.construct.simple import AffineNonlinear
from breze.arch.construct import sparsefiltering
from breze.arch.util import ParameterSet
from breze.learn.base import (
    UnsupervisedModel, TransformBrezeWrapperMixin)


class SparseFiltering(UnsupervisedModel,
                      TransformBrezeWrapperMixin):

    transform_expr_name = 'output'

    def __init__(self, n_inpt, n_output, feature_transfer='softabs',
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create a SparseFiltering object.

        Parameters
        ----------

        n_inpt : int
            Input dimensionality of the data.

        n_output: int
            Dimensionality of the hidden feature dimension.

        feature_transfer : string or callable
            Transfer function to use. If a string referring any function found
            in ``breze.arch.component.transfer`` or a function that given an
            (n, d) array returns an (n, d) array as theano expressions.

        max_iter : int
            Maximum number of optimization iterations to perform.

        verbose : bool
            Flag indicating whether to print out information during fitting.
        """
        self.n_inpt = n_inpt
        self.n_output = n_output
        self.feature_transfer = feature_transfer

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        self.use_imp_weight = False

        self._init_exprs()

    def _init_exprs(self):
        inpt = T.matrix('inpt')
        if theano.config.compute_test_value:
            inpt.tag.test_value = np.empty((2, self.n_inpt))

        P = self.parameters = ParameterSet()

        self.layer = AffineNonlinear(inpt, self.n_inpt, self.n_output,
                                     'identity', declare=P.declare,
                                     use_bias=False)

        self.loss_layer = sparsefiltering.SparseFilteringLoss(
            self.layer.output, self.feature_transfer)

        super(SparseFiltering, self).__init__(
            inpt=inpt, output=self.layer.output, loss=self.loss_layer.total,
            parameters=P)
