# -*- coding: utf-8 -*-

"""Module for learning various types of recurrent networks."""


import itertools 

import breze.model.sequential.rnn as rnn
import climin
import numpy as np
import theano.tensor as T


class Rnn(rnn.RecurrentNetwork):

    def __init__(self, n_inpt, n_hidden, n_output, 
                 hidden_transfer='tanh', out_transfer='identity', 
                 loss='squared', pooling=None,
                 optimizer='ksd',
                 pretraining=False,
                 max_iter=1000,
                 verbose=False):
        super(Rnn, self).__init__(
            n_inpt, n_hidden, n_output, hidden_transfer, out_transfer,
            loss, pooling)
        self.optimizer = optimizer
        self.pretraining = pretraining
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

    def iter_fit(self, X, Z):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the returned
        iterator. The model is in a valid state after each iteration, so that
        the optimization can be broken any time by the caller.
        
        :param X: A (t, n ,d) array where _t_ is the number of time steps,
            _n_ is the number of data samples and _d_ is the dimensionality of
            a data sample at a single time step.
        :param Z: A (t, n, l) array where _t_ and _n_ are defined as in _X_,
            but _l_ is the dimensionality of the output sequences at a single
            time step.
        """
        if self.pretraining:
            raise NotImplementedError('pretraining not implemented') 

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
        for i in itr:
            pass

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

