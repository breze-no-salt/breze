# -*- coding: utf-8 -*-

"""Module for learning various types of recurrent networks."""


import itertools 

import breze.model.sequential.rnn as rnn
import climin
import numpy as np
import theano.tensor as T

from breze.model.neural import TwoLayerPerceptron


class Rnn(rnn.RecurrentNetwork):
    """Class implementing recurrent neural networks for supervised learning..
    
    The class inherits from breze's RecurrentNetwork class and adds several
    sklearn like methods.
    """

    def __init__(self, n_inpt, n_hidden, n_output, 
                 hidden_transfer='tanh', out_transfer='identity', 
                 loss='squared', pooling=None,
                 optimizer='ksd',
                 pretrain=False,
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
        :param pretrain: Number of pretrain iterations to do. This will perform
            training locally, i.e. with all recurrent connections set to 0 and
            not applying any updates to them.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(Rnn, self).__init__(
            n_inpt, n_hidden, n_output, hidden_transfer, out_transfer,
            loss, pooling)
        self.optimizer = optimizer
        self.pretrain = pretrain
        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None
        self.parameters.data[:] = np.random.standard_normal(self.parameters.data.shape)

    def _gauss_newton_product(self):
        """Return a theano expression for the product of the networks
        Gauss-Newton matrix with an arbitrary vector."""

        # Shortcuts.
        output_in = self.exprs['output-in']
        loss = self.exprs['loss']
        flat_pars = self.parameters.flat
        hidden_in_rec = self.exprs['hidden-in-rec']
        p = self.exprs['some-vector'] = T.vector('some-vector')

        Jp = T.Rop(output_in, flat_pars, p)
        HJp = T.grad(T.sum(T.grad(loss, output_in) * Jp),
                 output_in, consider_constant=[Jp])
        Hp = T.grad(T.sum(HJp * output_in),
                    flat_pars, consider_constant=[HJp, Jp])

        return Hp

    def _d_loss(self):
        """Return a theano expression for the gradient of the loss wrt the
        flat parameters of the model."""
        return T.grad(self.exprs['loss'], self.parameters.flat)

    def _make_loss_functions(self):
        """Return triple (f_loss, f_d_loss, f_Hp) of functions.
        
         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
         - f_Hp returns the product of an arbitrary vector of the Gauss Newton
           matrix of the loss.
        """
        d_loss = self._d_loss()
        Hp = self._gauss_newton_product()

        f_loss = self.function(['inpt', 'target'], 'loss', explicit_pars=True)
        f_d_loss = self.function(['inpt', 'target'], d_loss, explicit_pars=True)
        f_Hp = self.function(['some-vector', 'inpt', 'target'], Hp,
                             explicit_pars=True)
        return f_loss, f_d_loss, f_Hp

    def _make_predict_functions(self):
        """Return a function to predict targets from input sequences."""
        return self.function(['inpt'], 'output')

    def _pretrain(self, X, Z):
        # Construct an MLP of same dimensions.
        net = TwoLayerPerceptron(
            self.n_inpt, self.n_hidden, self.n_output,
            self.hidden_transfer, self.out_transfer, self.loss)
        common_pars = ('in_to_hidden', 'hidden_bias', 
                       'hidden_to_out', 'out_bias')

        # Copy parameters to mlp.
        for p in common_pars:
            net.parameters[p][:] = self.parameters[p]

        # Create loss functions.
        d_loss_wrt_pars = T.grad(net.exprs['loss'], net.parameters.flat)
        f_loss = net.function(['inpt', 'target'], 'loss',
                              explicit_pars=True)
        f_d_loss = net.function(['inpt', 'target'], d_loss_wrt_pars,
                                explicit_pars=True)

        # Disentangle sequence data.
        X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        Z = Z.reshape((Z.shape[0] * Z.shape[1], Z.shape[2]))
        args = (([X, Z], {}) for _ in itertools.count())
        opt = climin.Lbfgs(net.parameters.data, f_loss, f_d_loss, args=args)

        # Train for some epochs with LBFGS.
        for i, info in enumerate(opt):
            loss = f_loss(net.parameters.data, X, Z)
            print 'pretrain', i, loss / X.size
            if i + 1 == self.pretrain:
                break

        # Copy parameters back.
        for p in common_pars:
            self.parameters[p][:] = net.parameters[p]

        # Set recurrent weights to 0.
        self.parameters['hidden_to_hidden'][:] = 0.

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
        if self.pretrain:
            self._pretrain(X, Z)

        f_loss, f_d_loss, f_Hp = self._make_loss_functions()

        args = itertools.repeat(([X, Z], {}))
        if self.optimizer == 'ksd':
            opt = climin.KrylovSubspaceDescent(
                self.parameters.data, f_loss, f_d_loss, f_Hp, 50,
                args=args)
        elif self.optimizer == 'rprop':
            opt = climin.Rprop(self.parameters.data, f_loss, f_d_loss,
                args=args)
        else:
            raise ValueError('unknown optimizer %s' % self.optimizer)

        for i, info in enumerate(opt):
            loss = info.get('loss', None)
            if loss is None:
                loss = f_loss(self.parameters.data, X, Z)
            info['loss'] = loss
            yield info

    def fit(self, X, Z):
        """Fit the parameters of the model to the given data with the
        given error function.

        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        :param Z: A (t, n, l) array where _t_ and _n_ are defined as in _X_,
            but _l_ is the dimensionality of the output sequences at a single
            time step.
        """
        itr = self.iter_fit(X, Z)
        for i, info in enumerate(itr):
            if i + 1 >= self.max_iter:
                break

    def predict(self, X):
        """Return the prediction of the network given input sequences.

        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        :returns: A (t, n, l) array where _t_ and _n_ are defined as in _X_,
            but _l_ is the dimensionality of the output sequences at a single
            time step.
        """
        if self.f_predict is None:
            self.f_predict = self._make_predict_functions()
        return self.f_predict(X)
