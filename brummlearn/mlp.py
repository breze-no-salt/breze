# -*- coding: utf-8 -*-

"""Module for learning various types of multilayer perceptrons."""


import itertools

import climin
import climin.util
import numpy as np
import theano.tensor as T
import theano.tensor.shared_randomstreams

from breze.model.neural import MultiLayerPerceptron

from brummlearn.base import SupervisedBrezeWrapperBase


class Mlp(MultiLayerPerceptron, SupervisedBrezeWrapperBase):

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 optimizer='lbfgs',
                 batch_size=-1,
                 max_iter=1000, verbose=False):
        """Create an Mlp object.

        This implementation uses a stack of affine mappings with a subsequent
        non linearity.

        :param n_inpt: Dimensionality of a single input.
        :param n_hiddens: List of integers, where each integer specifies the
            size of that layer.
        :param n_output: Dimensionality of a single output.
        :param hidden_transfers: List of transfer functions. A transfer function
            is either a string pointing to a function in
            ``breze.component.transfer`` or a function taking a theano 2D tensor
            and returning a tensor of the same shape.
        :param optimizer: String identifying the optimizer to use. Can only be
            ``lbfgs`` for now.
        :param batch_size: Number of examples per batch when calculing the loss
            and its derivatives. -1 means to use all samples every time.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
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


def truncate(arr, max_sqrd_length, axis):
    if arr.ndim != 2 or axis not in (0, 1):
        raise ValueError('only 2d arrays allowed')

    sqrd_lengths = (arr**2).sum(axis=axis)
    too_big_by = sqrd_lengths / max_sqrd_length
    divisor = np.sqrt(too_big_by)
    non_violated = sqrd_lengths < max_sqrd_length
    divisor[np.where(non_violated)] = 1.

    if axis == 0:
        divisor = divisor[np.newaxis, :]
    else:
        divisor = divisor[:, np.newaxis]
    arr /= divisor


class DropoutMlp(Mlp):

    #
    # TODO
    #
    #  - making schedule of steprate and momentum right -- for now, it is
    #    changed each mini batch, not epoch.
    #

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 p_dropout_inpt=.2, p_dropout_hidden=.5,
                 max_norm=15,
                 optimizer=None,
                 batch_size=-1,
                 max_iter=1000, verbose=False):

        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hidden = p_dropout_hidden
        self.max_norm = max_norm

        if optimizer is None:
            optimizer = self.standard_optimizer()

        super(DropoutMlp, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer,
            loss=loss, optimizer=optimizer, batch_size=batch_size,
            max_iter=max_iter, verbose=verbose)

        self.parameters.data[:] = np.random.normal(0, 0.01,
            self.parameters.data.shape)

    def standard_optimizer(self):
        """Return a dictionary suitable for climin.util.optimizer which
        specifies the standard optimizer for dropout mlps."""
        steprate_0 = 10
        steprate_decay = 0.998
        momentum_0 = 0.5
        momentum_equilibrium = 0.99
        n_momentum_anneal_steps = 500

        def repeater(iter, n):
          for i in iter:
            for j in range(n):
              yield i

        momentum_inc = (momentum_equilibrium - momentum_0) / n_momentum_anneal_steps
        momentum_annealed = (momentum_0 + momentum_inc * t
                             for t in range(1, n_momentum_anneal_steps))
        momentum = itertools.chain(momentum_annealed,
                                   itertools.repeat(momentum_equilibrium))

        # The steprate depends on the momentum. Thus we define it again to
        # consume it for the step rate iterator.
        momentum_annealed2 = (momentum_0 + momentum_inc * t
                              for t in range(1, n_momentum_anneal_steps))
        momentum2 = itertools.chain(momentum_annealed2,
                                    itertools.repeat(momentum_equilibrium))

        steprate = ((1 - m) * steprate_0 * steprate_decay**t
                    for t, m in enumerate(momentum2))

        steprate = repeater(steprate, 500)
        momentum = repeater(momentum, 500)

        return 'gd', {
            'steprate': steprate,
            'momentum': momentum,
        }

    def _make_loss_functions(self, mode='FAST_RUN'):
        """Return pair (f_loss, f_d_loss) of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        rng = T.shared_randomstreams.RandomStreams()

        # Drop out inpts.
        inpt = self.exprs['inpt']
        inpt_dropout = rng.binomial(inpt.shape, p=self.p_dropout_inpt)
        inpt_dropped_out = inpt_dropout * inpt
        givens = {inpt: inpt_dropped_out}
        loss = theano.clone(self.exprs['loss'], givens)

        n_layers = len(self.n_hiddens)
        for i in range(n_layers - 1):
            # Drop out hidden.
            hidden = self.exprs['hidden_%i' % i]
            hidden_dropout = rng.binomial(hidden.shape, p=self.p_dropout_hidden)
            # TODO check out which one of those two is faster.
            #hiddens_dropped_out = T.switch(dropout, 0, hiddens)
            hidden_dropped_out = hidden_dropout * hidden
            givens = {hidden: hidden_dropped_out}
            loss = theano.clone(loss, givens)

        d_loss = T.grad(loss, self.parameters.flat)

        f_loss = self.function(['inpt', 'target'], loss, explicit_pars=True,
                               mode=mode)
        f_d_loss = self.function(['inpt', 'target'], d_loss, explicit_pars=True,
                                 mode=mode)
        return f_loss, f_d_loss

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
            truncate(W, self.max_norm, axis=0)

            n_layers = len(self.n_hiddens)
            for i in range(n_layers - 1):
                W = self.parameters['hidden_to_hidden_%i' % i]
                truncate(W, self.max_norm, axis=0)

    def predict(self, X):
        pass
