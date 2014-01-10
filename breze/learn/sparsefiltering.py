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

from breze.arch.model.feature import sparsefiltering
from breze.arch.model import linear
from breze.arch.util import ParameterSet, Model
from breze.learn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin)


class SparseFiltering(Model, UnsupervisedBrezeWrapperBase,
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

        super(SparseFiltering, self).__init__()

    def _init_pars(self):
        spec = sparsefiltering.parameters(self.n_inpt, self.n_output)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
        }
        P = self.parameters

        self.exprs.update(linear.exprs(
            self.exprs['inpt'], P.in_to_out, 0, 'identity'))
        self.exprs.update(sparsefiltering.loss(self.exprs['output'], self.feature_transfer))

