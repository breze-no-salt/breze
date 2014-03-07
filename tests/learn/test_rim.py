# -*- coding: utf-8 -*-

import numpy as np

from breze.learn.rim import Rim
from breze.learn.utils import theano_floatx


def test_rim_fit():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    rim = Rim(2, 10, 0.1, max_iter=10)
    rim.fit(X)


def test_rim_iter_fit():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    rim = Rim(2, 10, 0.1, max_iter=10)
    for i, info in enumerate(rim.iter_fit(X)):
        if i >= 10:
            break


def test_rim_transform():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    rim = Rim(2, 10, 0.1, max_iter=10)
    rim.transform(X)
