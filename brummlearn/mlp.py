# -*- coding: utf-8 -*-

"""Module for learning various types of multilayer perceptrons."""


import itertools

import climin
import climin.util
import climin.gd
from climin.project import max_length_columns

import numpy as np
import theano.tensor as T
import theano.tensor.shared_randomstreams

from breze.model.neural import MultiLayerPerceptron
from breze.model import fastdropout
from breze.component import corrupt
from brummlearn.base import SupervisedBrezeWrapperBase


class Mlp(MultiLayerPerceptron, SupervisedBrezeWrapperBase):
    """Multilayer perceptron class.

    This implementation uses a stack of affine mappings with a subsequent
    non linearity each.

    Parameters
    ----------

    n_inpt : integer
        Dimensionality of a single input.

    n_hiddens : list of integers
        List of ``k`` integers, where ``k`` is thenumber of layers. Each gives
        the size of the corresponding layer.

    n_output : integer
        Dimensionality of a single output.

    hidden_transfers : list, each item either string or function
        Transfer functions for each of the layers. Can be either a string which
        is then used to look up a transfer function in
        ``breze.component.transfer`` or a function that given a Theano tensor
        returns a tensor of the same shape.

    out_transfer : string or function
        Either a string to look up a function in ``breze.component.transfer`` or
        a function that given a Theano tensor returns a tensor of the same
        shape.

    optimizer : string, pair
        Argument is passed to ``climin.util.optimizer`` to construct an
        optimizer.

    batch_size : integer, None
        Number of examples per batch when calculting the loss
        and its derivatives. None means to use all samples every time.

    max_iter : int
        Maximum number of optimization iterations to perform. Only respected
        during``.fit()``, not ``.iter_fit()``.

    verbose : boolean
        Flag indicating whether to print out information during fitting.
    """

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 optimizer='lbfgs',
                 batch_size=None,
                 max_iter=1000, verbose=False):
        super(Mlp, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer,
            loss)

        self.optimizer = optimizer
        self.batch_size = batch_size

        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape)


def dropout_optimizer_conf(
        steprate_0=1, steprate_decay=0.998, momentum_0=0.5,
        momentum_eq=0.99, n_momentum_anneal_steps=500,
        n_repeats=500):
    """Return a dictionary suitable for climin.util.optimizer which
    specifies the standard optimizer for dropout mlps."""
    steprate = climin.gd.decaying(steprate_0, steprate_decay)
    momentum = climin.gd.linear_annealing(
        momentum_0, momentum_eq, n_momentum_anneal_steps)

    # Define another time for steprate calculcation.
    momentum2 = climin.gd.linear_annealing(
        momentum_0, momentum_eq, n_momentum_anneal_steps)
    steprate = ((1 - j) * i for i, j in itertools.izip(steprate, momentum2))

    steprate = climin.gd.repeater(steprate, n_repeats)
    momentum = climin.gd.repeater(momentum, n_repeats)

    return 'gd', {
        'steprate': steprate,
        'momentum': momentum,
    }


class DropoutMlp(Mlp):

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 p_dropout_inpt=.2, p_dropout_hidden=.5,
                 max_length=15,
                 optimizer=None,
                 batch_size=-1,
                 max_iter=1000, verbose=False):

        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hidden = p_dropout_hidden
        self.max_length = max_length

        if optimizer is None:
            optimizer = dropout_optimizer_conf()

        super(DropoutMlp, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer,
            loss=loss, optimizer=optimizer, batch_size=batch_size,
            max_iter=max_iter, verbose=verbose)

        self.parameters.data[:] = np.random.normal(
            0, 0.01, self.parameters.data.shape)

    def _make_loss_functions(self, mode=None):
        """Return pair (f_loss, f_d_loss) of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        rng = T.shared_randomstreams.RandomStreams()

        # Drop out inpts.
        inpt = self.exprs['inpt']
        inpt_dropped_out = corrupt.mask(inpt, self.p_dropout_inpt, rng)
        givens = {inpt: inpt_dropped_out}
        loss = theano.clone(self.exprs['loss'], givens)

        n_layers = len(self.n_hiddens)
        for i in range(n_layers - 1):
            # Drop out hidden.
            hidden = self.exprs['hidden_%i' % i]
            hidden_dropped_out = corrupt.mask(hidden, self.p_dropout_hidden, rng)
            givens = {hidden: hidden_dropped_out}
            loss = theano.clone(loss, givens)

        d_loss = T.grad(loss, self.parameters.flat)

        f_loss = self.function(['inpt', 'target'], loss, explicit_pars=True,
                               mode=mode)
        f_d_loss = self.function(['inpt', 'target'], d_loss, explicit_pars=True,
                                 mode=mode)
        return f_loss, f_d_loss

    # TODO: wrong docstring
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

        args = self._make_args(X, Z)
        opt = self._make_optimizer(f_loss, f_d_loss, args)

        for i, info in enumerate(opt):
            yield info
            W = self.parameters['in_to_hidden']
            max_length_columns(W, self.max_length)

            n_layers = len(self.n_hiddens)
            for i in range(n_layers - 1):
                W = self.parameters['hidden_to_hidden_%i' % i]
                max_length_columns(W, self.max_length)
            W = self.parameters['hidden_to_out']
            max_length_columns(W, self.max_length)


class FastDropoutNetwork(fastdropout.FastDropoutNetwork,
                         SupervisedBrezeWrapperBase):

    # TODO: dropout rates have to be strictly positive, otherwise there is a
    # non positive variance.
    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 optimizer='lbfgs',
                 batch_size=None,
                 p_dropout_inpt=.2,
                 p_dropout_hidden=.5,
                 max_length=15,
                 inpt_var=0,
                 var_bias_offset=0.0,
                 max_iter=1000, verbose=False):
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hidden = p_dropout_hidden
        self.max_length = max_length
        self.inpt_var = inpt_var
        self.var_bias_offset = var_bias_offset

        super(FastDropoutNetwork, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer,
            loss)
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape)

    def iter_fit(self, X, Z):
        for info in super(FastDropoutNetwork, self).iter_fit(X, Z):
            yield info
            W = self.parameters['in_to_hidden']
            max_length_columns(W, self.max_length)

            n_layers = len(self.n_hiddens)
            for i in range(n_layers - 1):
                W = self.parameters['hidden_to_hidden_%i' % i]
                max_length_columns(W, self.max_length)
            W = self.parameters['hidden_to_out']
            max_length_columns(W, self.max_length)
