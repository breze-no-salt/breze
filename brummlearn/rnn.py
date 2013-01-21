# -*- coding: utf-8 -*-

"""Module for learning various types of recurrent networks."""


import itertools

import breze.model.sequential.rnn as rnn
import climin
import climin.stops
import numpy as np
import theano
import theano.tensor as T

from breze.model.neural import TwoLayerPerceptron

from brummlearn.base import (
    SupervisedBrezeWrapperBase, UnsupervisedBrezeWrapperBase,
    TransformBrezeWrapperMixin)


class BaseRnn(object):

    # TODO: default loss should not be squared, which makes reordering ofi
    # arguments necessary.

    def __init__(self, n_inpt, n_hidden, n_output,
                 hidden_transfer='tanh', out_transfer='identity',
                 loss='squared', pooling=None,
                 optimizer='rprop',
                 batch_size=None,
                 max_iter=1000,
                 verbose=False):
        """Create and return a ``Rnn`` object.

        :param n_inpt: Number of inputs per time step to the network.
        :param n_hidden: Size of the hidden state.
        :param n_output: Size of the output of the network.
        :param hidden_transfer: Transfer function to use for the network. This
            can either (a) be a string and reference a transfer function from
            ``breze.component.transfer`` or a function which takes a theano
            tensor3 and returns a tensor of equal size.
        :param out_transfer: Output function to use for the network. This
            can either (a) be a string and reference a transfer function from
            ``breze.component.transfer`` or a function which takes a theano
            tensor3 and returns a tensor of equal size.
        :param loss: Loss which is going to be optimized. This can either be a
            string and reference a loss function found in
            ``breze.component.distance`` or a function which takes two theano
            tensors (one being the output of the network, the other some target)
            and returns a theano scalar.
        :param pooling: One of ``sum``, ``mean``, ``prod``, ``min``, ``max`` or
            ``None``. If not None, the output is pooled over the time dimension,
            essentially making the network return a tensor2 instead of a
            tensor3.
        :param optimizer: Either ``ksd`` referring to KrylovSubspaceDescent or
            ``rprop``.
        :param batch_size: Number of examples per batch when calculing the loss
            and its derivatives. None means to use all samples every time.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(BaseRnn, self).__init__(
            n_inpt, n_hidden, n_output, hidden_transfer, out_transfer,
            loss, pooling)
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None
        self.parameters.data[:] = np.random.standard_normal(self.parameters.data.shape)

    def _gauss_newton_product(self):
        """Return a theano expression for the product of the networks
        Gauss-Newton matrix with an arbitrary vector."""

        # Shortcuts.
        output_in = self.exprs['output_in']
        loss = self.exprs['loss']
        flat_pars = self.parameters.flat
        p = self.exprs['some-vector'] = T.vector('some-vector')

        Jp = T.Rop(output_in, flat_pars, p)
        HJp = T.grad(T.sum(T.grad(loss, output_in) * Jp),
            output_in, consider_constant=[Jp])
        Hp = T.grad(T.sum(HJp * output_in),
                    flat_pars, consider_constant=[HJp, Jp])

        return Hp

    def _make_loss_functions(self):
        """Return pair `f_loss, f_d_loss` of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
           matrix of the loss.
        """
        d_loss = self._d_loss()

        args = list(self.data_arguments)
        f_loss = self.function(args, 'loss', explicit_pars=True)
        f_d_loss = self.function(args, d_loss, explicit_pars=True)
        return f_loss, f_d_loss


class SupervisedRnn(BaseRnn, rnn.SupervisedRecurrentNetwork,
                    SupervisedBrezeWrapperBase):
    """Class implementing recurrent neural networks for supervised learning..

    The class inherits from breze's RecurrentNetwork class and adds several
    sklearn like methods.
    """

    def __init__(self, n_inpt, n_hidden, n_output,
                 hidden_transfer='tanh', out_transfer='identity',
                 loss='squared', pooling=None,
                 optimizer='rprop',
                 batch_size=None,
                 max_iter=1000,
                 verbose=False):
        if pooling is None:
            self.sample_dim = 1, 1
        else:
            self.sample_dim = 1, 0
        super(SupervisedRnn, self).__init__(
            n_inpt, n_hidden, n_output, hidden_transfer, out_transfer, loss,
            pooling, optimizer, batch_size, max_iter, verbose)

    def iter_fit(self, X, Z):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the returned
        iterator. The model is in a valid state after each iteration, so that
        the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.

        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        :param Z: A (t, n, l) array where _t_ and _n_ are defined as in _X_,
            but _l_ is the dimensionality of the output sequences at a single
            time step.
        """
        f_loss, f_d_loss = self._make_loss_functions()

        args = itertools.repeat(([X, Z], {}))
        opt = self._make_optimizer(f_loss, f_d_loss, args)

        for i, info in enumerate(opt):
            yield info


class UnsupervisedRnn(BaseRnn, rnn.UnsupervisedRecurrentNetwork,
                      UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin):
    """Class implementing recurrent neural networks for unsupervised learning..

    The class inherits from breze's RecurrentNetwork class and adds several
    sklearn like methods.
    """

    transform_expr_name = 'output'
    sample_dim = 1,

    def iter_fit(self, X):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the returned
        iterator. The model is in a valid state after each iteration, so that
        the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.

        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        """
        f_loss, f_d_loss = self._make_loss_functions()

        args = self._make_args(X)
        opt = self._make_optimizer(f_loss, f_d_loss, args)

        for i, info in enumerate(opt):
            yield info


class BaseLstm(BaseRnn):

    def __init__(self, n_inpt, n_hidden, n_output,
                 hidden_transfer='tanh', out_transfer='identity',
                 loss='squared', pooling=None,
                 optimizer='rprop',
                 max_iter=1000,
                 verbose=False):
        """Create and return a ``Rnn`` object.

        :param n_inpt: Number of inputs per time step to the network.
        :param n_hidden: Size of the hidden state.
        :param n_output: Size of the output of the network.
        :param hidden_transfer: Transfer function to use for the network. This
            can either (a) be a string and reference a transfer function from
            ``breze.component.transfer`` or a function which takes a theano
            tensor3 and returns a tensor of equal size.
        :param out_transfer: Output function to use for the network. This
            can either (a) be a string and reference a transfer function from
            ``breze.component.transfer`` or a function which takes a theano
            tensor3 and returns a tensor of equal size.
        :param loss: Loss which is going to be optimized. This can either be a
            string and reference a loss function found in
            ``breze.component.distance`` or a function which takes two theano
            tensors (one being the output of the network, the other some target)
            and returns a theano scalar.
        :param pooling: One of ``sum``, ``mean``, ``prod``, ``min``, ``max`` or
            ``None``. If not None, the output is pooled over the time dimension,
            essentially making the network return a tensor2 instead of a
            tensor3.
        :param optimizer: Either ``ksd`` referring to KrylovSubspaceDescent or
            ``rprop``.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(BaseLstm, self).__init__(n_inpt, n_hidden, n_output,
            hidden_transfer, out_transfer, loss, pooling, optimizer, False,
            max_iter, verbose)


class SupervisedLstm(BaseLstm, rnn.SupervisedLstmRecurrentNetwork,
                     SupervisedBrezeWrapperBase):
    """Class implementing recurrent neural networks with LSTM cells for
    supervised learning.

    The class inherits from breze's RecurrentNetwork class and adds several
    sklearn like methods.
    """
    # TODO fix docstring

    def iter_fit(self, X, Z):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the returned
        iterator. The model is in a valid state after each iteration, so that
        the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.

        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        :param Z: A (t, n, l) array where _t_ and _n_ are defined as in _X_,
            but _l_ is the dimensionality of the output sequences at a single
            time step.
        """

        f_loss, f_d_loss = self._make_loss_functions()

        args = itertools.repeat(([X, Z], {}))
        opt = self._make_optimizer(f_loss, f_d_loss, args)

        for i, info in enumerate(opt):
            yield info


class UnsupervisedLstm(BaseLstm, rnn.UnsupervisedLstmRecurrentNetwork,
                       UnsupervisedBrezeWrapperBase,
                       TransformBrezeWrapperMixin):
    """Class implementing recurrent neural networks with LSTM cells for
    unsupervised learning.

    The class inherits from breze's RecurrentNetwork class and adds several
    sklearn like methods.
    """
    # TODO fix docstring
    transform_expr_name = 'output'

    def iter_fit(self, X):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the returned
        iterator. The model is in a valid state after each iteration, so that
        the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.

        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        """

        f_loss, f_d_loss = self._make_loss_functions()


        args = itertools.repeat(([X], {}))
        opt = self._make_optimizer(f_loss, f_d_loss, args)

        for i, info in enumerate(opt):
            yield info
