# -*- coding: utf-8 -*-

import numpy as np

from brummlearn.mlp import Mlp


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
    Z = np.random.standard_normal((10, 1))
    mlp = Mlp(2, [10], 1, ['tanh'], 'identity', 'squared', max_iter=10)
    mlp.predict(X)
