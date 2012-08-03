# -*- coding: utf-8 -*-

import numpy as np

from brummlearn.rnn import Rnn


def test_rnn_fit():
    X = np.random.standard_normal((10, 5, 2))
    Z = np.random.standard_normal((10, 5, 3))
    rnn = Rnn(2, 10, 3, max_iter=10)
    rnn.fit(X, Z)


def test_rnn_iter_fit():
    X = np.random.standard_normal((10, 5, 2))
    Z = np.random.standard_normal((10, 5, 3))
    rnn = Rnn(2, 10, 3, max_iter=10)
    for i, info in enumerate(rnn.iter_fit(X, Z)):
        if i >= 10:
            break


def test_rnn_predict():
    X = np.random.standard_normal((10, 5, 2))
    Z = np.random.standard_normal((10, 5, 3))
    rnn = Rnn(2, 10, 3, max_iter=10)
    rnn.predict(X)
