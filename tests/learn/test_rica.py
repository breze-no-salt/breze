# -*- coding: utf-8 -*-

import numpy as np

from breze.learn.rica import Rica
from breze.learn.utils import theano_floatx


def test_rica_fit():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)

    rica = Rica(2, 10, code_transfer='softabs',
                hidden_transfer='identity', loss='squared',
                c_ica=0.5, max_iter=10)
    rica.fit(X)


def test_rica_iter_fit():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    rica = Rica(2, 10, code_transfer='softabs',
                hidden_transfer='identity', loss='squared',
                c_ica=0.5, max_iter=10)
    for i, info in enumerate(rica.iter_fit(X)):
        if i >= 10:
            break


def test_rica_transform():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    rica = Rica(2, 10, code_transfer='softabs',
                hidden_transfer='identity', loss='squared',
                c_ica=0.5, max_iter=10)
    rica.transform(X)


def test_rica_reconstruct():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    rica = Rica(2, 10, code_transfer='softabs',
                hidden_transfer='identity', loss='squared',
                c_ica=0.5, max_iter=10)
    rica.reconstruct(X)
