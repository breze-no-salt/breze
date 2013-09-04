# -*- coding: utf-8 -*-

import numpy as np

from brummlearn.sparsecoding import SparseCoding


def test_sparse_coding_fit():
    X = np.random.standard_normal((10, 2))
    sf = SparseCoding(2, 7, max_iter=10)
    sf.fit(X)


def test_sparse_coding_iter_fit():
    X = np.random.standard_normal((10, 2))
    sf = SparseCoding(2, 7, max_iter=10)
    for i, info in enumerate(sf.iter_fit(X)):
        if i >= 10:
            break


def test_sparse_coding_transform():
    X = np.random.standard_normal((10, 2))
    sf = SparseCoding(2, 7, max_iter=10)
    sf.transform(X)
