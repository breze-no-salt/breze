# -*- coding: utf-8 -*-

import numpy as np

from breze.learn.sparsecoding import SparseCoding
from breze.learn.utils import theano_floatx

from nose.plugins.skip import SkipTest


def test_sparse_coding_fit():
    raise SkipTest()
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)

    sf = SparseCoding(2, 7, max_iter=10)
    sf.fit(X)


def test_sparse_coding_iter_fit():
    raise SkipTest()
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    sf = SparseCoding(2, 7, max_iter=10)
    for i, info in enumerate(sf.iter_fit(X)):
        if i >= 10:
            break


def test_sparse_coding_transform():
    raise SkipTest()
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)
    sf = SparseCoding(2, 7, max_iter=10)
    sf.transform(X)
