# -*- coding: utf-8 -*-
import numpy as np

from breze.learn.cnn import Cnn
from breze.learn.utils import theano_floatx
import theano

theano.config.exception_verbosity = 'high'


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
            pool_shapes=[(2, 2), (2, 2)],
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
            pool_shapes=[(2, 2), (2, 2)],
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
            pool_shapes=[(2, 2), (2, 2)],
            filter_shapes=[(4, 4), (3, 3)],
            )
    m.predict(X)