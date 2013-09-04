# -*- coding: utf-8 -*-

"""Regularized Information Maximization.

As introduced in

  Discriminative clustering by regularized information maximization},
  by Gomes, R. and Krause, A. and Perona, P.},
  NIPS 2010
"""

import numpy as np
import theano

from breze.model.rim import Rim as _Rim
from brummlearn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin)


class Rim(_Rim, UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin):

    transform_expr_name = 'output'

    def __init__(self, n_inpt, n_clusters, c_rim, optimizer='lbfgs',
                 max_iter=1000, verbose=False):
        """Create a Rim object.

        :param n_inpt: Input dimensionality of the data.
        :param n_clusters: Amount of clusters to use.
        :param c_rim: Float value indicating the regularization strength.
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
        super(Rim, self).__init__(n_inpt, n_clusters,
            c_rim)
        self.f_transform = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose
