# -*- coding: utf-8 -*-

import itertools

import numpy as np
import theano
import theano.tensor as T

from climin import Lbfgs

from breze.model.neural import MultilayerPerceptron as MLP


# Make data.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=theano.config.floatX)
Z = np.array([[0], [1], [1], [0]], dtype=theano.config.floatX)
args = itertools.repeat(([X, Z], {}))


# Make network.
net = MLP(2, 10, 1, 'sigmoid', 'sigmoid', 'bernoulli_cross_entropy')
f = net.function(['inpt', 'target'], 'loss', explicit_pars=True)
d_loss_wrt_pars = T.grad(net.exprs['loss'], net.parameters.flat)
fprime = net.function(['inpt', 'target'], d_loss_wrt_pars, explicit_pars=True)

net.parameters.data[:] = np.random.standard_normal(net.parameters.data.shape)


# Optimize!
opt = Lbfgs(net.parameters.data, f, fprime, args=args)
for i, info in enumerate(opt):
    print 'loss', f(net.parameters.data, X, Z)
