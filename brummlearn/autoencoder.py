# -*- coding: utf-8 -*-

"""Autoencoder."""

import numpy as np

from breze.model.feature import (
    AutoEncoder as _AutoEncoder,
    SparseAutoEncoder as _SparseAutoEncoder,
    ContractiveAutoEncoder as _ContractiveAutoEncoder)
from brummlearn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)


class AutoEncoder(_AutoEncoder, UnsupervisedBrezeWrapperBase,
                  TransformBrezeWrapperMixin, ReconstructBrezeWrapperMixin):

    transform_expr_name = 'hidden'

    def __init__(self, n_inpt, n_hidden, hidden_transfer='identity',
                 out_transfer='identity', loss='squared', tied_weights=True,
                 batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create an AutoEncoder object.

        :param n_inpt: Input dimensionality of the data.
        :param n_hidden: Dimensionality of the hidden feature dimension.
        :param hidden_transfer: Transfer function to use for the hidden units.
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
        :param tied_weights: Flag indicating whether to use tied weights.
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
        super(AutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer, loss, tied_weights)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.f_transform = None
        self.f_reconstruct = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape)
        self.max_iter = max_iter
        self.verbose = verbose


class SparseAutoEncoder(_SparseAutoEncoder, UnsupervisedBrezeWrapperBase,
                        TransformBrezeWrapperMixin,
                        ReconstructBrezeWrapperMixin):

    transform_expr_name = 'hidden'

    def __init__(self, n_inpt, n_hidden, hidden_transfer='sigmoid',
                 out_transfer='identity', reconstruct_loss='squared',
                 c_sparsity=1, sparsity_loss='neg_cross_entropy',
                 sparsity_target=0.01,
                 tied_weights=True, batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create a SparseAutoEncoder object.

        :param n_inpt: Input dimensionality of the data.
        :param n_hidden: Dimensionality of the hidden feature dimension.
        :param hidden_transfer: Transfer function to use for the hidden units.
            Can be a string referring any function found in
            ``breze.component.transfer`` or a function that given an (n, d)
            array returns an (n, d) array as theano expressions.
        :param out_transfer: Output transfer function of the linear auto encoder
            for calculation of the reconstruction cost.
        :param reconstruct_loss: Reconstruction part of the loss which is
            going to be optimized. This can either be a string and reference a
            loss function found in ``breze.component.distance`` or a function
            which takes two theano tensors (one being the output of the network,
            the other some target) and returns a theano scalar.
        :param c_sparsity: Coefficient weighing the sparsity cost in comparison
            to the reconstruction cost.
        :param sparsity_loss: Sparsity part of the loss which is
            going to be optimized. This can either be a string and reference a
            loss function found in ``breze.component.distance`` or a function
            which takes two theano tensors (one being the output of the network,
            the other some target) and returns a theano scalar.
        :param sparsity_target: Subtract this value from each hidden unit before
            applying the sparsity loss.
        :param tied_weights: Flag indicating whether to use tied weights.
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
        super(SparseAutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer,
            reconstruct_loss, c_sparsity, sparsity_loss, sparsity_target,
            tied_weights)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.f_transform = None
        self.f_reconstruct = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape)
        self.max_iter = max_iter
        self.verbose = verbose


class ContractiveAutoEncoder(_ContractiveAutoEncoder,
                             UnsupervisedBrezeWrapperBase,
                             TransformBrezeWrapperMixin,
                             ReconstructBrezeWrapperMixin):

    transform_expr_name = 'hidden'

    def __init__(self, n_inpt, n_hidden, hidden_transfer='sigmoid',
                 out_transfer='identity', reconstruct_loss='squared',
                 c_jacobian=1, tied_weights=True, batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create a ContractiveAutoEncoder object.

        :param n_inpt: Input dimensionality of the data.
        :param n_hidden: Dimensionality of the hidden feature dimension.
        :param hidden_transfer: Transfer function to use for the hidden units.
            Can be a string referring any function found in
            ``breze.component.transfer`` or a function that given an (n, d)
            array returns an (n, d) array as theano expressions.
        :param out_transfer: Output transfer function of the linear auto encoder
            for calculation of the reconstruction cost.
        :param reconstruct_loss: Reconstruction part of the loss which is
            going to be optimized. This can either be a string and reference a
            loss function found in ``breze.component.distance`` or a function
            which takes two theano tensors (one being the output of the network,
            the other some target) and returns a theano scalar.
        :param c_jacobian: Coefficient weighing the Jacobian cost in comparison
            to the reconstruction cost.
        :param tied_weights: Flag indicating whether to use tied weights.
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
        super(ContractiveAutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer,
            reconstruct_loss, c_jacobian,
            tied_weights)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.f_transform = None
        self.f_reconstruct = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape)
        self.max_iter = max_iter
        self.verbose = verbose
