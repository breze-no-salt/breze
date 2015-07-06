# -*- coding: utf-8 -*-

import cPickle

import numpy as np

from breze.learn.mlp import Mlp
from breze.learn.mlp import DropoutMlp
from breze.learn.mlp import FastDropoutNetwork

from breze.arch.component.loss import squared
from breze.learn.utils import theano_floatx


def test_mlp_pickle():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))

    X, Z = theano_floatx(X, Z)

    mlp = Mlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10)

    itr = mlp.iter_fit(X, Z)
    itr.next()

    Y = mlp.predict(X)

    pickled = cPickle.dumps(mlp)
    mlp2 = cPickle.loads(pickled)

    Y2 = mlp2.predict(X)

    assert np.allclose(Y, Y2)


def test_mlp_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))

    X, Z = theano_floatx(X, Z)

    mlp = Mlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10)
    mlp.fit(X, Z)


def test_mlp_fit_with_imp_weight():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    W = np.random.random((10, 1)) > 0.5

    X, Z, W = theano_floatx(X, Z, W)

    mlp = Mlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10,
              imp_weight=True)
    mlp.fit(X, Z, W)


def test_mlp_iter_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    X, Z = theano_floatx(X, Z)

    mlp = Mlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10)
    for i, info in enumerate(mlp.iter_fit(X, Z)):
        if i >= 10:
            break


def test_mlp_predict():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    mlp = Mlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10)
    mlp.predict(X)


def test_dmlp_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    X, Z = theano_floatx(X, Z)

    mlp = DropoutMlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10,
                     p_dropout_inpt=.2, p_dropout_hiddens=[0.5])
    mlp.fit(X, Z)


def test_dmlp_iter_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    X, Z = theano_floatx(X, Z)

    mlp = DropoutMlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10,
                     p_dropout_inpt=.2, p_dropout_hiddens=[0.5])
    for i, info in enumerate(mlp.iter_fit(X, Z)):
        if i >= 10:
            break


def test_dmlp_predict():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)

    mlp = DropoutMlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10,
                     p_dropout_inpt=.2, p_dropout_hiddens=[0.5])
    mlp.predict(X)


def test_fd_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    X, Z = theano_floatx(X, Z)
    loss = lambda target, prediction: squared(
        target, prediction[:, :target.shape[1]])
    mlp = FastDropoutNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    mlp.fit(X, Z)


def test_fd_iter_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    X, Z = theano_floatx(X, Z)
    loss = lambda target, prediction: squared(
        target, prediction[:, :target.shape[1]])
    mlp = FastDropoutNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    for i, info in enumerate(mlp.iter_fit(X, Z)):
        if i >= 10:
            break


def test_fd_predict():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    loss = lambda target, prediction: squared(
        target, prediction[:, :target.shape[1]])
    mlp = FastDropoutNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    mlp.predict(X)
