# -*- coding: utf-8 -*-

import copy

import numpy as np

from nose.tools import with_setup

from breze.learn import sgvb
from breze.learn.utils import theano_floatx
from breze.utils.testhelpers import use_test_values


class MyVAE(sgvb.VariationalAutoEncoder,
            sgvb.MlpGaussLatentVAEMixin,
            sgvb.MlpBernoulliVisibleVAEMixin):
    pass


class MyFDVAE(sgvb.FastDropoutVariationalAutoEncoder,
              sgvb.FastDropoutMlpGaussLatentVAEMixin,
              sgvb.FastDropoutMlpGaussVisibleVAEMixin):
    pass


class MyStorn(sgvb.StochasticRnn,
              sgvb.GaussLatentStornMixin,
              sgvb.GaussVisibleStornMixin):
    pass


@with_setup(*use_test_values('raise'))
def test_vae():
    X = np.random.random((2, 10))
    X, = theano_floatx(X)

    m = MyVAE(
        10, [20, 30], 4, [15, 25],
        ['tanh'] * 2, ['rectifier'] * 2,
        optimizer='rprop', batch_size=None,
        max_iter=3)

    m.fit(X)
    m.score(X)
    m.estimate_nll(X[:2], 2)


@with_setup(*use_test_values('raise'))
def test_vae_imp_weight():
    X = np.random.random((2, 10))
    W = np.random.random((2, 1))
    X, W = theano_floatx(X, W)

    m = MyVAE(
        10, [20, 30], 4, [15, 25],
        ['tanh'] * 2, ['rectifier'] * 2,
        optimizer='rprop', batch_size=None,
        max_iter=3,
        use_imp_weight=True)

    m.fit(X, W)
    m.score(X, W)


@with_setup(*use_test_values('raise'))
def test_storn():
    X = np.random.random((3, 3, 2))
    X, = theano_floatx(X)

    kwargs = {
        'n_inpt': X.shape[2],
        'n_hiddens_recog': [5],
        'n_latent': 11,
        'n_hiddens_gen': [7],
        'recog_transfers': ['tanh'],
        'gen_transfers': ['rectifier'],
        'p_dropout_inpt': .1,
        'p_dropout_hiddens': [.1],
        'p_dropout_shortcut': [.1],
        'p_dropout_hidden_to_out': .1,
        'use_imp_weight': False,
        'optimizer': 'adam',
        'batch_size': None,
        'verbose': False,
        'max_iter': 3,
    }

    m = MyStorn(**kwargs)

    print 'fitting'
    m.fit(X)
    print 'scoring'
    m.score(X)

    print 'initializing'
    m.initialize()


@with_setup(*use_test_values('raise'))
def test_storn_sampling():
    X = np.random.random((3, 5, 2))
    X, = theano_floatx(X)

    m = MyStorn(
        2, [5], 17, [5],
        ['tanh'] * 1, ['rectifier'] * 1,
        optimizer='rprop', batch_size=None,
        max_iter=3)

    m.parameters.data[...] = 1

    print 'sampling with prefix'
    m.sample(5, visible_map=True, prefix=X[:, :1, :])


def test_storn_copy():
    X = np.random.random((3, 5, 2))
    X, = theano_floatx(X)

    m = MyStorn(
        2, [5], 17, [5],
        ['tanh'] * 1, ['rectifier'] * 1,
        optimizer='rprop', batch_size=None,
        max_iter=3)

    m.parameters.data[...] = 1

    m2 = copy.deepcopy(m)

    print dir(m)
    print dir(m2)

    print '---'

    print m.__dict__
    print m2.__dict__

    assert hasattr(m2, 'exprs')


def test_vae_copy():
    X = np.random.random((2, 10))
    X, = theano_floatx(X)

    m = MyVAE(
        10, [20, 30], 4, [15, 25],
        ['tanh'] * 2, ['rectifier'] * 2,
        optimizer='rprop', batch_size=None,
        max_iter=3,
        use_imp_weight=True)

    m2 = copy.deepcopy(m)

    print dir(m)
    print dir(m2)

    print '---'

    print m.__dict__
    print m2.__dict__

    assert hasattr(m2, 'exprs')


@with_setup(*use_test_values('raise'))
def test_deep_fdvae():
    X = np.random.random((2, 10))
    X, = theano_floatx(X)

    m = MyFDVAE(
        95, [20, 30], 4, [15, 25],
        ['rectifier'] * 2, ['rectifier'] * 2,
        optimizer='rprop', batch_size=None,
        max_iter=3)
