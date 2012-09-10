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


class DropoutMlp(Mlp):

    #
    # TODO
    #
    # 1. Droping out of inputs,
    # 2. making schedule of steprate and momentum right -- for now, it is
    #    changed each mini batch, not epoch.
    #

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 p_dropout_inpt=.2, p_dropout_hidden=.5,
                 max_norm=15,
                 optimizer=None,
                 batch_size=-1,
                 max_iter=1000, verbose=False):
        if len(n_hiddens) > 1:
            raise ValueError('DropoutMlp not ready for multiple hidden layers')

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
        n_momentum_anneal_steps = 1000

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

        return 'gd', {
            'steprate': steprate,
            'momentum': momentum,
        }

    def _make_loss_functions(self, mode='FAST_RUN'):
        """Return pair (f_loss, f_d_loss) of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        hiddens = self.exprs['hidden_0']
        rng = T.shared_randomstreams.RandomStreams()
        dropout = rng.binomial(hiddens.shape, p=self.p_dropout_hidden)

        # TODO check out which one of those two is faster.
        #hiddens_dropped_out = T.switch(dropout, 0, hiddens)
        hiddens_dropped_out = dropout * hiddens

        givens = {hiddens: hiddens_dropped_out}
        loss = theano.clone(self.exprs['loss'], givens)
        d_loss = T.grad(loss, self.parameters.flat)

        f_loss = self.function(['inpt', 'target'], loss, explicit_pars=True,
                               mode=mode)
        f_d_loss = self.function(['inpt', 'target'], d_loss, explicit_pars=True,
                                 mode=mode)
        return f_loss, f_d_loss

    def normalize_weights(self, W):
        weight_norms = (W**2).sum(axis=1)
        non_violated = (weight_norms < self.max_norm)
        divisor = weight_norms[:, np.newaxis]
        divisor[np.where(non_violated), :] = 1.
        W /= divisor

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
            self.normalize_weights(self.parameters['in_to_hidden'])
            #self.normalize_weights(self.parameters['hidden_to_out'])

    def predict(self, X):
        pass
