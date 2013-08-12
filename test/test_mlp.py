# -*- coding: utf-8 -*-

import numpy as np

from brummlearn.mlp import Mlp, FastDropoutNetwork

from breze.component.loss import squared


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
