# -*- coding: utf-8 -*-

import random

import numpy as np
import theano

from breze.learn.gaussianprocess import GaussianProcess
from breze.learn.utils import theano_floatx

from nose.plugins.skip import SkipTest


def test_gp_fit():
    X = np.arange(-2, 2, .1)[:, np.newaxis].astype(theano.config.floatX)
    X, = theano_floatx(X)
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 20)
    X = X[idxs]
    Z = np.sin(X)

    X, Z = theano_floatx(X, Z)

    gp = GaussianProcess(1, max_iter=10, kernel='ardse')
    gp.fit(X, Z)


def test_gp_fit_linear():
    X = np.arange(-2, 2, .1)[:, np.newaxis].astype(theano.config.floatX)
    X, = theano_floatx(X)
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 20)
    X = X[idxs]
    Z = np.sin(X)

    X, Z = theano_floatx(X, Z)

    gp = GaussianProcess(1, max_iter=10, kernel='linear')
    gp.fit(X, Z)


def test_gp_iter_fit():
    X = np.arange(-2, 2, .1)[:, np.newaxis].astype(theano.config.floatX)
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 20)
    X = X[idxs]
    Z = np.sin(X)
    X, Z = theano_floatx(X, Z)

    gp = GaussianProcess(1, max_iter=10, kernel='ardse')
    for i, info in enumerate(gp.iter_fit(X, Z)):
        if i >= 10:
            break


def test_gp_predict_linear():
    raise SkipTest()
    X = np.arange(-2, 2, .1)[:, np.newaxis].astype(theano.config.floatX)
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 200)
    X = X[idxs]
    Z = np.sin(X)
    Z += np.random.normal(0, 1e-1, X.shape)
    X, Z = theano_floatx(X, Z)

    gp = GaussianProcess(1, max_iter=1, kernel='linear')
    gp.fit(X, Z)
    print gp.predict(X)


def test_gp_predict_matern52():
    X = np.arange(-2, 2, .1)[:, np.newaxis].astype(theano.config.floatX)
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 20)
    X = X[idxs]
    Z = np.sin(X)
    Z += np.random.normal(0, 1e-1, X.shape)
    X, Z = theano_floatx(X, Z)

    gp = GaussianProcess(1, max_iter=10, kernel='matern52')
    gp.fit(X, Z)
    print gp.predict(X)


def test_gp_predict_maxrows():
    X = np.arange(-2, 2, .1)[:, np.newaxis].astype(theano.config.floatX)
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 6)
    X = X[idxs]
    Z = np.sin(X)
    Z += np.random.normal(0, 1e-1, X.shape).astype(theano.config.floatX)
    X, Z = theano_floatx(X, Z)

    gp = GaussianProcess(1, max_iter=10, kernel='matern52')
    gp.fit(X, Z)
    Y = gp.predict(X)
    Y2 = gp.predict(X, max_rows=2)

    assert np.allclose(Y, Y2)


def test_gp_sample_parameters():
    X = np.arange(-2, 2, .1)[:, np.newaxis].astype(theano.config.floatX)
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 10)
    X = X[idxs]
    Z = np.sin(X)
    Z += np.random.normal(0, 1e-1, X.shape).astype(theano.config.floatX)
    X, Z = theano_floatx(X, Z)

    gp = GaussianProcess(1, max_iter=1, kernel='ardse')
    gp.store_dataset(X, Z)
    gp.sample_parameters()
    print gp.predict(X, True)
