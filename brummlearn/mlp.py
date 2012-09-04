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
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(Mlp, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer,
            loss)

        self.optimizer = optimizer

        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape)


class DropoutMlp(Mlp):

    def __init__(self, n_inpt, n_hiddens, n_output, 
                 hidden_transfers, out_transfer, loss,
                 p_dropout_inpt=.2, p_dropout_hidden=.5,
                 max_iter=1000, verbose=False):
        if len(n_hiddens) > 1:
            raise ValueError('DropoutMlp not ready for multiple hidden layers')
        super(DropoutMlp, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer,
            loss, max_iter, verbose)


    def _make_loss_functions(self):
        """Return pair (f_loss, f_d_loss) of functions.
        
         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
        """
        d_loss = self._d_loss()
        hiddens = self.exprs['hidden_0']
        rng = T.shared_randomstreams.RandomStreams()
        dropout = rng.binomial(hiddens.shape)
        hiddens_dropped_out = hiddens * dropout

        givens = {hiddens: hiddens_dropped_out}
        loss = theano.clone(self.exprs['loss'], givens)
        d_loss = theano.clone(d_loss, givens)

        f_loss = self.function(['inpt', 'target'], loss, explicit_pars=True)
        f_d_loss = self.function(['inpt', 'target'], d_loss, explicit_pars=True)
        return f_loss, f_d_loss

    def _make_optimizer(self, f, fprime, args):
        # TODO make this configurable from the outside.
        steprate_0 = 0.1
        steprate_decay = 0.998
        momentum_0 = 0.5
        momentum_equilibrum = 0.99 
        n_momentum_anneal_steps = 1000

        steprate = (steprate0 * steprate_decay**t for t in itertools.count(1))
        momentum = itertools.chain(
            (momentum_0 + 
             min(momentum_equilibrum, 
                (momentum_equilibrum - momentum_0) / n_momentum_anneal_steps)
             * t for t in itertools.count(1)),
            itertools.repeat(momentum_equilibrum))

        # TODO dropout inputs

        return climin.GradientDescent(
            self.parameters.data, fprime, 
            steprate=steprate, momentum=momentum, args=args)

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
            loss = info.get('loss', None)
            if loss is None:
                loss = f_loss(self.parameters.data, X, Z)
            info['loss'] = loss
            yield info

    def predict(self, X):
        pass
