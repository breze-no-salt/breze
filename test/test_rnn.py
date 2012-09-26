# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T

from brummlearn.rnn import (
    SupervisedRnn, UnsupervisedRnn,
    SupervisedLstm, UnsupervisedLstm)


def test_srnn_fit():
    X = np.random.standard_normal((10, 5, 2))
    Z = np.random.standard_normal((10, 5, 3))
    rnn = SupervisedRnn(2, 10, 3, max_iter=10)
    rnn.fit(X, Z)


def test_srnn_iter_fit():
    X = np.random.standard_normal((10, 5, 2))
    Z = np.random.standard_normal((10, 5, 3))
    rnn = SupervisedRnn(2, 10, 3, max_iter=10)
    for i, info in enumerate(rnn.iter_fit(X, Z)):
        if i >= 10:
            break


def test_srnn_predict():
    X = np.random.standard_normal((10, 5, 2))
    rnn = SupervisedRnn(2, 10, 3, max_iter=10)
    rnn.predict(X)


def test_usrnn_fit():
    X = np.random.standard_normal((10, 5, 2))
    rnn = UnsupervisedRnn(2, 10, 3, loss=lambda x: T.log(x.sum()), max_iter=10)
    rnn.fit(X)


def test_usrnn_iter_fit():
    X = np.random.standard_normal((10, 5, 2))
    rnn = UnsupervisedRnn(2, 10, 3, loss=lambda x: T.log(x.sum()), max_iter=10)
    for i, info in enumerate(rnn.iter_fit(X)):
        if i >= 10:
            break


def test_usrnn_transform():
    X = np.random.standard_normal((10, 5, 2))
    rnn = UnsupervisedRnn(2, 10, 3, loss=lambda x: T.log(x.sum()), max_iter=10)
    rnn.transform(X)


def test_slstm():
    X = np.random.standard_normal((10, 5, 2))
    Z = np.random.standard_normal((10, 5, 3))
    rnn = SupervisedLstm(2, 10, 3, max_iter=10)
    rnn.fit(X, Z)


def test_slstm_iter_fit():
    X = np.random.standard_normal((10, 5, 2))
    Z = np.random.standard_normal((10, 5, 3))
    rnn = SupervisedLstm(2, 10, 3, max_iter=10)
    for i, info in enumerate(rnn.iter_fit(X, Z)):
        if i >= 10:
            break


def test_slstm_predict():
    X = np.random.standard_normal((10, 5, 2))
    rnn = SupervisedLstm(2, 10, 3, max_iter=10)
    rnn.predict(X)


def test_uslstm_fit():
    X = np.random.standard_normal((10, 5, 2))
    rnn = UnsupervisedLstm(2, 10, 3, loss=lambda x: T.log(x.sum()), max_iter=10)
    rnn.fit(X)


def test_uslstm_iter_fit():
    X = np.random.standard_normal((10, 5, 2))
    rnn = UnsupervisedLstm(2, 10, 3, loss=lambda x: T.log(x.sum()), max_iter=10)
    for i, info in enumerate(rnn.iter_fit(X)):
        if i >= 10:
            break


def test_uslstm_transform():
    X = np.random.standard_normal((10, 5, 2))
    rnn = UnsupervisedLstm(2, 10, 3, loss=lambda x: T.log(x.sum()), max_iter=10)
    rnn.transform(X)
