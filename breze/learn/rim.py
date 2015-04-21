# -*- coding: utf-8 -*-

"""Regularized Information Maximization.

As introduced in [1]_.

References
----------

  .. [1] Discriminative clustering by regularized information maximization,
     by Gomes, R. and Krause, A. and Perona, P., NIPS 2010
"""

import numpy as np
import theano
import theano.tensor as T

from breze.arch.model import rim
from breze.arch.util import ParameterSet, Model
from breze.learn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin)
from breze.arch.model import linear, rim


class Rim(Model, UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin):
    """Class for regularized information maximization.

    Attributes
    ----------

    parameters : ParamterSet object
        Parameters of the model.

    n_inpt : integer
        Input dimensionality of the data.

    n_cluster : integer
        Amount of clusters to use.

    c_rim : float
        Value indicating the regularization strength.

    optimizer : string or pair
        Can be either a string or a pair. In any case,
        ``climin.util.optimizer`` is used to construct an optimizer. In the
        case of a string, the string is used as an identifier for the
        optimizer which is then instantiated with default arguments. If a
        pair, expected to be ``(identifier, kwargs)`` for more fine control
        of the optimizer.

    max_iter : integer
        Maximum number of optimization iterations to perform.

    verbose : boolean
        Flag indicating whether to print out information during fitting."""

    transform_expr_name = 'output'

    def __init__(self, n_inpt, n_cluster, c_rim, optimizer='lbfgs',
                 max_iter=1000, verbose=False):
        """Create a Rim object.

        Parameters
        ----------

        n_inpt : integer
            Input dimensionality of the data.

        n_cluster : integer
            Amount of clusters to use.

        c_rim : float
            Value indicating the regularization strength.

        optimizer : string or pair
            Can be either a string or a pair. In any case,
            ``climin.util.optimizer`` is used to construct an optimizer. In the
            case of a string, the string is used as an identifier for the
            optimizer which is then instantiated with default arguments. If a
            pair, expected to be ``(identifier, kwargs)`` for more fine control
            of the optimizer.

        max_iter : integer
            Maximum number of optimization iterations to perform.

        verbose : boolean
            Flag indicating whether to print out information during fitting.
        """
        self.n_inpt = n_inpt
        self.n_cluster = n_cluster
        self.c_rim = c_rim

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        super(Rim, self).__init__()

    def _init_pars(self):
        spec = linear.parameters(self.n_inpt, self.n_cluster)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
        }
        P = self.parameters

        self.exprs.update(linear.exprs(
            self.exprs['inpt'], P.in_to_out, P.bias, 'softmax'))
        self.exprs.update(rim.loss(self.exprs['output'], [P.in_to_out], self.c_rim))
