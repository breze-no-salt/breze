# -*- coding: utf-8 -*-

import random

import numpy as np

from breze.learn.gaussianprocess import GaussianProcess


def test_gp_fit():
    X = np.arange(-2, 2, .01)[:, np.newaxis]
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 200)
    X = X[idxs]
    Z = np.sin(X)

    gp = GaussianProcess(1, max_iter=10)
    gp.fit(X, Z)


def test_gp_iter_fit():
    X = np.arange(-2, 2, .01)[:, np.newaxis]
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 200)
    X = X[idxs]
    Z = np.sin(X)

    gp = GaussianProcess(1, max_iter=10)
    for i, info in enumerate(gp.iter_fit(X, Z)):
        if i >= 10:
            break


def test_gp_predict_linear():
    X = np.arange(-2, 2, .01)[:, np.newaxis]
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 200)
    X = X[idxs]
    Z = np.sin(X)
    Z += np.random.normal(0, 1e-1, X.shape)

    gp = GaussianProcess(1, max_iter=1, kernel='linear')
    gp.fit(X, Z)
    print gp.predict(X)


def test_gp_predict_matern52():
    X = np.arange(-2, 2, .01)[:, np.newaxis]
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 200)
    X = X[idxs]
    Z = np.sin(X)
    Z += np.random.normal(0, 1e-1, X.shape)

    gp = GaussianProcess(1, max_iter=10, kernel='matern52')
    gp.fit(X, Z)
    print gp.predict(X)


def test_gp_predict_maxrows():
    X = np.arange(-2, 2, .01)[:, np.newaxis]
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 6)
    X = X[idxs]
    Z = np.sin(X)
    Z += np.random.normal(0, 1e-1, X.shape)

    gp = GaussianProcess(1, max_iter=10, kernel='matern52')
    gp.fit(X, Z)
    Y = gp.predict(X)
    Y2 = gp.predict(X, max_rows=2)

    assert np.allclose(Y, Y2)


def test_gp_sample_parameters():
    X = np.arange(-2, 2, .01)[:, np.newaxis]
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 20)
    X = X[idxs]
    Z = np.sin(X)
    Z += np.random.normal(0, 1e-1, X.shape)

    gp = GaussianProcess(1, max_iter=1, kernel='linear')
    gp.store_dataset(X, Z)
    gp.sample_parameters()
    print gp.predict(X, True)
