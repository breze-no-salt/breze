# -*- coding: utf-8 -*-

import copy

import numpy as np
import theano

from breze.learn import sgvb
from breze.learn.utils import theano_floatx
from breze.utils.testhelpers import use_test_values


def test_vae():
    X = np.random.random((2, 10))
    X, = theano_floatx(X)

    m = sgvb.VariationalAutoEncoder(
        10, [20, 30], 4, [15, 25],
        ['tanh'] * 2, ['rectifier'] * 2,
        assumptions=Assmptn(),
        optimizer='rprop', batch_size=None,
        max_iter=3)

    m.fit(X)
    m.score(X)
    m.transform(X)
    m.estimate_nll(X[:2], 2)


def test_vae_imp_weight():
    X = np.random.random((2, 10))
    W = np.random.random((2, 1))
    X, W = theano_floatx(X, W)

    theano.config.compute_test_value = 'raise'

    m = sgvb.VariationalAutoEncoder(
        10, [20, 30], 4, [15, 25],
        ['tanh'] * 2, ['rectifier'] * 2,
        assumptions=Assmptn(),
        optimizer='rprop', batch_size=None,
        max_iter=3,
        use_imp_weight=True)

    m.fit(X, W)
    m.score(X, W)
    m.transform(X)


@use_test_values('raise')
def test_storn():
    X = np.random.random((3, 3, 2))
    X, = theano_floatx(X)

    class MyStorn(sgvb.StochasticRnn,
                  sgvb.GaussLatentStornMixin,
                  sgvb.GaussVisibleStornMixin):
        pass

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


def test_storn_sampling():
    theano.config.compute_test_value = 'raise'
    X = np.random.random((3, 5, 2))
    X, = theano_floatx(X)

    class Assmptn(sgvb.DiagGaussLatentAssumption, sgvb.DiagGaussVisibleAssumption):
        pass

    m = sgvb.StochasticRnn(
        2, [5], 17, [5],
        ['tanh'] * 1, ['rectifier'] * 1,
        assumptions=Assmptn(),
        optimizer='rprop', batch_size=None,
        max_iter=3)

    m.parameters.data[...] = 1

    print 'sampling with prefix'
    m.sample(5, visible_map=True, prefix=X[:, :1, :])

    #m._sample_one_step(
    #    np.empty(5), np.empty(5),
    #    np.empty((1, 1, 2)),
    #    np.empty((1, 1, 17)))


    #P = m.parameters

    #initial_means = [P[i.recurrent.initial_mean]
    #                for i in m.vae.gen.hidden_layers]
    #initial_stds = [P[i.recurrent.initial_std]
    #                for i in m.vae.gen.hidden_layers]

    #from breze.learn.sgvb import flatten_list
    #args = flatten_list(zip(initial_means, initial_stds))

    #inpt = np.zeros((1, 1, 2))
    #latent_samples = 1 / np.zeros((1, 1, 17))
    #args += [inpt, latent_samples[:1]]

    #s1 = m._sample_one_step_vmap(*args)
    #print s1
    #s2 = m._sample_one_step_vmap(*args)
    #pen


def test_storn_copy():
    theano.config.compute_test_value = 'raise'
    X = np.random.random((3, 5, 2))
    X, = theano_floatx(X)

    class Assmptn(sgvb.DiagGaussLatentAssumption, sgvb.DiagGaussVisibleAssumption):
        pass

    m = sgvb.StochasticRnn(
        2, [5], 17, [5],
        ['tanh'] * 1, ['rectifier'] * 1,
        assumptions=Assmptn(),
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

    class Assmptn(sgvb.DiagGaussLatentAssumption, sgvb.DiagGaussVisibleAssumption):
        pass

    m = sgvb.VariationalAutoEncoder(
        10, [20, 30], 4, [15, 25],
        ['tanh'] * 2, ['rectifier'] * 2,
        assumptions=Assmptn(),
        optimizer='rprop', batch_size=None,
        max_iter=3)

    m2 = copy.deepcopy(m)

    print dir(m)
    print dir(m2)

    print '---'

    print m.__dict__
    print m2.__dict__

    assert hasattr(m2, 'exprs')


