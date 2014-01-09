# -*- coding: utf-8 -*-

import numpy as np

from breze.learn.mlp import Mlp
from breze.learn.mlp import DropoutMlp
#from breze.learn.mlp import FastDropoutNetwork
#from breze.learn.mlp import AwnNetwork

from breze.arch.component.loss import squared


def test_mlp_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    mlp = Mlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10)
    mlp.fit(X, Z)


def test_mlp_iter_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    mlp = Mlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10)
    for i, info in enumerate(mlp.iter_fit(X, Z)):
        if i >= 10:
            break


def test_mlp_predict():
    X = np.random.standard_normal((10, 2))
    mlp = Mlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10)
    mlp.predict(X)


def test_dmlp_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    mlp = DropoutMlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10,
                     p_dropout_inpt=.2, p_dropout_hiddens=[0.5])
    mlp.fit(X, Z)


def test_dmlp_iter_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    mlp = DropoutMlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10,
                     p_dropout_inpt=.2, p_dropout_hiddens=[0.5])
    for i, info in enumerate(mlp.iter_fit(X, Z)):
        if i >= 10:
            break


def test_dmlp_predict():
    X = np.random.standard_normal((10, 2))
    mlp = DropoutMlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10,
                     p_dropout_inpt=.2, p_dropout_hiddens=[0.5])
    mlp.predict(X)


def test_fd_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    loss = lambda target, prediction: squared(target, prediction[:, :target.shape[1]])
    mlp = FastDropoutNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    mlp.fit(X, Z)


def test_fd_iter_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    loss = lambda target, prediction: squared(target, prediction[:, :target.shape[1]])
    mlp = FastDropoutNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    for i, info in enumerate(mlp.iter_fit(X, Z)):
        if i >= 10:
            break


def test_fd_predict():
    X = np.random.standard_normal((10, 2))
    loss = lambda target, prediction: squared(target, prediction[:, :target.shape[1]])
    mlp = FastDropoutNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    mlp.predict(X)


def test_awn_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    loss = lambda target, prediction: squared(target, prediction[:, :target.shape[1]])
    mlp = AwnNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    mlp.fit(X, Z)


def test_awn_iter_fit():
    X = np.random.standard_normal((10, 2))
    Z = np.random.standard_normal((10, 1))
    loss = lambda target, prediction: squared(target, prediction[:, :target.shape[1]])
    mlp = AwnNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    for i, info in enumerate(mlp.iter_fit(X, Z)):
        if i >= 10:
            break


def test_awn_predict():
    X = np.random.standard_normal((10, 2))
    loss = lambda target, prediction: squared(target, prediction[:, :target.shape[1]])
    mlp = AwnNetwork(
        2, [10], 1, ['rectifier'], 'identity', loss, max_iter=10)
    mlp.predict(X)
