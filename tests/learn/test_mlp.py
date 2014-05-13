# -*- coding: utf-8 -*-

import numpy as np

from breze.learn.mlp import Mlp
from breze.learn.mlp import DropoutMlp
from breze.learn.cnn import Cnn
from breze.learn.mlp import FastDropoutNetwork

from breze.arch.component.loss import squared
from breze.learn.utils import theano_floatx


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

    mlp = Mlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10, imp_weight=True)
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


def test_cnn_iter_fit():
    X = np.random.standard_normal((10, 2 * 100 * 50))
    Z = np.random.random((10, 1)) > 0.5
    X, Z = theano_floatx(X, Z)

    m = Cnn(100 * 50, [10, 15], [20, 12], 1,
            ['sigmoid', 'sigmoid'], ['rectifier', 'rectifier'],
            'sigmoid',
            'cat_ce', 100, 50, 2,
            optimizer=('rmsprop', {'step_rate': 1e-4, 'decay': 0.9}),
            batch_size=2,
            max_iter=10,
            pool_size=(2, 2),
            filter_shapes=[(4, 4), (3, 3)],
            )
    for i, info in enumerate(m.iter_fit(X, Z)):
        if i >= 10:
            break


def test_cnn_fit():
    X = np.random.standard_normal((10, 2 * 100 * 50))
    Z = np.random.random((10, 1)) > 0.5
    X, Z = theano_floatx(X, Z)

    m = Cnn(100 * 50, [10, 15], [20, 12], 1,
            ['sigmoid', 'sigmoid'], ['rectifier', 'rectifier'],
            'sigmoid',
            'cat_ce', 100, 50, 2,
            optimizer=('rmsprop', {'step_rate': 1e-4, 'decay': 0.9}),
            batch_size=2,
            max_iter=10,
            pool_size=(2, 2),
            filter_shapes=[(4, 4), (3, 3)],
            )
    m.fit(X, Z)


def test_cnn_predict():
    X = np.random.standard_normal((10, 2 * 100 * 50))
    X, = theano_floatx(X)

    m = Cnn(100 * 50, [10, 15], [20, 12], 1,
            ['sigmoid', 'sigmoid'], ['rectifier', 'rectifier'],
            'sigmoid',
            'cat_ce', 100, 50, 2,
            optimizer=('rmsprop', {'step_rate': 1e-4, 'decay': 0.9}),
            batch_size=2,
            max_iter=10,
            pool_size=(2, 2),
            filter_shapes=[(4, 4), (3, 3)],
            )
    m.predict(X)


def test_fd_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    X, Z = theano_floatx(X, Z)
    loss = lambda target, prediction: squared(target, prediction[:, :target.shape[1]])
    mlp = FastDropoutNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    mlp.fit(X, Z)


def test_fd_iter_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    X, Z = theano_floatx(X, Z)
    loss = lambda target, prediction: squared(target, prediction[:, :target.shape[1]])
    mlp = FastDropoutNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    for i, info in enumerate(mlp.iter_fit(X, Z)):
        if i >= 10:
            break


def test_fd_predict():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    loss = lambda target, prediction: squared(target, prediction[:, :target.shape[1]])
    mlp = FastDropoutNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    mlp.predict(X)
