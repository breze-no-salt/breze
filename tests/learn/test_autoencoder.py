# -*- coding: utf-8 -*-

import numpy as np
from breze.learn import autoencoder
from breze.learn.utils import theano_floatx


def test_autoencoder():
    X = np.random.random((100, 10))
    X, = theano_floatx(X)

    m = autoencoder.AutoEncoder(10, [100], ['tanh'], 'identity', 'squared',
                                tied_weights=True, max_iter=10)
    m.fit(X)
    m.score(X)
    m.transform(X)


def test_deepautoencoder():
    X = np.random.random((100, 10))
    X, = theano_floatx(X)

    m = autoencoder.AutoEncoder(10, [100, 10, 100],
                                ['tanh', 'identity', 'tanh'], 'identity',
                                'squared',
                                tied_weights=False, max_iter=10)
    m.fit(X)
    m.score(X)
    m.transform(X)


def test_sparse_autoencoder():
    X = np.random.random((100, 10))
    X, = theano_floatx(X)

    m = autoencoder.SparseAutoEncoder(
        10, [100], ['sigmoid'], 'identity', 'squared', tied_weights=True,
        sparsity_target=0.01, c_sparsity=3., sparsity_loss='bern_bern_kl')
    m.fit(X)
    m.score(X)
    m.transform(X)


def test_contractive_autoencoder():
    X = np.random.random((100, 10))
    X, = theano_floatx(X)

    m = autoencoder.ContractiveAutoEncoder(
        10, [100], ['sigmoid'], 'identity', 'squared', tied_weights=True,
        c_jacobian=3, max_iter=10)

    m.fit(X)
    m.score(X)
    m.transform(X)


def test_denoising_autoencoder():
    X = np.random.random((100, 10))
    X, = theano_floatx(X)

    m = autoencoder.DenoisingAutoEncoder(
        10, [100], ['sigmoid'], 'identity', 'squared', tied_weights=True,
        noise_type='gauss', c_noise=.3, max_iter=10)

    m.fit(X)
    m.score(X)
    m.transform(X)
