# -*- coding: utf-8 -*-

import numpy as np

from breze.learn.sparsefiltering import SparseFiltering
from breze.learn.utils import theano_floatx


def test_sparse_filtering_fit():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)

    sf = SparseFiltering(2, 10, max_iter=10)
    sf.fit(X)


def test_sparse_filtering_iter_fit():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)

    sf = SparseFiltering(2, 10, max_iter=10)
    for i, info in enumerate(sf.iter_fit(X)):
        if i >= 10:
            break


def test_sparse_filtering_transform():
    X = np.random.standard_normal((10, 2))
    X, = theano_floatx(X)

    sf = SparseFiltering(2, 10, max_iter=10)
    sf.transform(X)
