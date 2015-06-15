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

from breze.arch.component.loss import sparse_filtering_loss
from breze.arch.model import linear
from breze.arch.util import ParameterSet
from breze.learn.base import (
    UnsupervisedModel, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)

from breze.arch.construct import neural, simple
from breze.learn.utils import theano_floatx

theano.config.compute_test_value = 'raise'


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

        self._init_exprs()


    def _init_exprs(self):
        inpt = T.matrix('inpt')
        inpt.tag.test_value, = theano_floatx(np.ones((3, self.n_inpt)))

        parameters = ParameterSet()

        self.affine = simple.AffineNonlinear(inpt, self.n_inpt, self.n_output, transfer='identity',
            declare=parameters.declare)

        output = self.affine.output

        sparse_loss = sparse_filtering_loss(output, self.feature_transfer)['loss']

        UnsupervisedModel.__init__(self, inpt=inpt,
                                 output=output,
                                 loss=sparse_loss,
                                 parameters=parameters)

        self.filters_in_to_hidden = parameters[self.affine.weights]
