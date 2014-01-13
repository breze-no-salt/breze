# -*- coding: utf-8 -*-

"""This module provides functionality for sparse coding.


Sparse coding is an algorithm to find low level representations of
data. The following loss is optimized:
:math:`||HW - X||^2_2 + \lambda ||H||_1`, where :math:`H` stands for the
representations, :math:`W` for a basis and :math:`X` for the data.

The optimization is performed with respect to both :math:`W` and :math:`H`.
This means that inferring new representations for so far unseed data is costly,
since an optimization has to be performed.
"""

import itertools

import climin.mathadapt as ma
import numpy as np
import theano
import theano.tensor as T

from breze.arch.model.feature import sparsecoding
from breze.arch.util import ParameterSet, Model
from breze.learn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin)


class SparseCoding(Model, UnsupervisedBrezeWrapperBase,
                   TransformBrezeWrapperMixin):
    """Simple implementation of sparse coding.

    The sparse coding loss is relaxed by using a soft version of the
    non-differentiable l1 norm: :math:`\sqrt{h^2 + \epsilon}`. This
    is then optimized with unconstrained optimization methods.

    Attributes
    ----------

    n_features : integer
        Amount of features to extract.

    c_sparsity : float
        Coefficient for the sparsity loss.

    optimizer : string or pair, optional [default: 'lbfgs']
        Can be either a string or a pair. In any case,
        ``climin.util.optimizer`` is used to construct an optimizer. In the
        case of a string, the string is used as an identifier for the
        optimizer which is then instantiated with default arguments. If a
        pair, expected to be (`identifier`, `kwargs`) for more fine control
        of the optimizer. This optimizer is used for finding codes.

    batch_size : integer
        Number of examples per batch when calculing the loss and its
        derivatives. None means to use all samples every time.

    max_iter : integer
        Maximum number of optimization iterations to perform. This refers
        to the amount of alternations between solving the least squares
        problem for the weight matrix and finding codes.

    parameters : ParameterSet object
        Contains the parameters of the model.
    """

    def __init__(self, n_inpt, n_feature, c_sparsity=5.,
                 optimizer='lbfgs', batch_size=None, max_iter=1000):
        """Create a SparseCoding object.

        Parameters
        ----------

        n_features : integer
            Amount of features to extract.

        c_sparsity : float
            Coefficient for the sparsity loss.

        optimizer : string or pair, optional [default: 'lbfgs']
            Can be either a string or a pair. In any case,
            ``climin.util.optimizer`` is used to construct an optimizer. In the
            case of a string, the string is used as an identifier for the
            optimizer which is then instantiated with default arguments. If a
            pair, expected to be (`identifier`, `kwargs`) for more fine control
            of the optimizer. This optimizer is used for finding codes.

        batch_size : integer
            Number of examples per batch when calculing the loss and its
            derivatives. None means to use all samples every time.

        max_iter : integer
            Maximum number of optimization iterations to perform. This refers
            to the amount of alternations between solving the least squares
            problem for the weight matrix and finding codes.
        """
        self.n_inpt = n_inpt
        self.n_feature = n_feature
        self.c_sparsity = c_sparsity

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter

        self.f_loss = None
        self.f_d_loss_wrt_feature = None

        super(SparseCoding, self).__init__()

    def _init_pars(self):
        spec = sparsecoding.parameters(self.n_feature, self.n_inpt)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.normal(
            0, 1e-5, self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
        }
        P = self.parameters
        self.exprs = sparsecoding.exprs(
            self.exprs['inpt'], P.feature_to_in, self.c_sparsity)

    def _make_loss_functions(self):
        if self.f_loss is None:
            self.f_loss = self.function(['feature_flat', 'inpt'], 'loss')
        if self.f_d_loss_wrt_feature is None:
            d_loss_wrt_feature = T.grad(self.exprs['loss'],
                                        self.exprs['feature_flat'])
            self.f_d_loss_wrt_feature = self.function(
                ['feature_flat', 'inpt'], d_loss_wrt_feature)

    def project_weights(self):
        w = self.parameters['feature_to_in']
        w /= ma.sqrt((w ** 2).sum(axis=0) + 1e-4)[np.newaxis]

    def fit(self, X):
        """Fit the parameters of the model.

        Parameters
        ----------

        X : array_like
            An array of shape `(n, d)` where `n` is the number of data points
            and `d` the input dimensionality."""
        for i, _ in enumerate(self.iter_fit(X)):
            if i == self.max_iter:
                break

    def powerfit(self, *args, **kwargs):
        raise NotImplementedError("no powerfit for this class available")

    def iter_fit(self, X):
        self._make_loss_functions()
        args = self._make_args(X)
        w = self.parameters['feature_to_in']

        for i in itertools.count():
            # Find new representations for the current data.
            (this_X,), _ = args.next()
            feature = self.transform(this_X)

            # Now solve for the best basis.
            w[...], _, _, _ = np.linalg.lstsq(feature, this_X)

            # Normalize it to prevent degenerate solutions.
            self.project_weights()

            yield {'n_iter': i,
                   'loss': self.f_loss(feature.ravel(), this_X)}

    def transform(self, X):
        """Transform data according to the model.

        Parameters
        ----------

        X : array_like
            An array of shape `(n, d)` where `n` is the number of data points
            and `d` the input dimensionality.

        Returns
        -------

        F : array_like
            An array of shape `(n, f)` where `n` is the number of samples and
            `f` is the number of features."""
        self._make_loss_functions()

        # This trick for initializing the features is due to the UFLDL tutorial
        # at http://ufldl.stanford.edu/wiki/.
        feature = np.dot(X, self.parameters['feature_to_in'].T)
        feature_flat = feature.ravel()

        args = itertools.repeat(((X,), {}))
        opt = self._make_optimizer(
            self.f_loss, self.f_d_loss_wrt_feature, args=args,
            wrt=feature_flat)

        for j, info in enumerate(opt):
            if j == 10:
                break
        return feature

    def inverse_transform(self, F):
        """Perform an inverse transformation of transformed data according to
        the model.

        Parameters
        ----------

        F : array_like
            An array of shape `(n, d)` where `n` is the number of data points
            and `d` the dimensionality if the feature space.

        Returns
        -------

        X : array_like
            An array of shape `(n, c)` where `n` is the number of samples and
            `c` is the dimensionality of the input space."""
        return np.dot(F, self.parameters['feature_to_in'])

    def reconstruct(self, X):
        """Reconstruct the data according to the model.

        Paramters
        ---------

        X : array_like
            An array of shape `(n, d)` where `n` is the number of data points
            and `d` the input dimensionality.

        Returns
        -------

        Y : array_like
            An array of shape `(n, d)` where `n` is the number of samples and
            `d` is the dimensionality of the input space."""
        F = self.transform(X)
        return self.inverse_transform(F)
