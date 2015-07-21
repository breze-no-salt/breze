# -*- coding: utf-8 -*-

import pickle

import numpy as np
import theano
import theano.tensor as T

from theano.gradient import jacobian
from nose.tools import with_setup

from breze.arch.construct.layer.varprop.sequential import FDRecurrent
from breze.arch.construct.layer.varprop.simple import AffineNonlinear

from breze.learn.rnn import (
    SupervisedFastDropoutRnn,
    SupervisedRnn)
from breze.learn.utils import theano_floatx
from breze.utils.testhelpers import use_test_values

from nose.plugins.skip import SkipTest


def test_srnn_fit():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    W = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)

    X, Z, W = theano_floatx(X, Z, W)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], max_iter=2)
    rnn.fit(X, Z)

    rnn = SupervisedRnn(
        2, [10], 3, hidden_transfers=['tanh'], max_iter=2, imp_weight=True)
    rnn.fit(X, Z, W)


@with_setup(*use_test_values('raise'))
def test_srnn_lstm_fit():
    X = np.random.standard_normal((13, 5, 4)).astype(theano.config.floatX)
    Z = np.random.standard_normal((13, 5, 3)).astype(theano.config.floatX)
    W = np.random.standard_normal((13, 5, 3)).astype(theano.config.floatX)

    X, Z, W = theano_floatx(X, Z, W)

    rnn = SupervisedRnn(4, [10], 3, hidden_transfers=['lstm'], max_iter=2)
    rnn.fit(X, Z)


@with_setup(*use_test_values('raise'))
def test_fdsrnn_lstm_fit():
    X = np.random.standard_normal((13, 5, 4)).astype(theano.config.floatX)
    Z = np.random.standard_normal((13, 5, 3)).astype(theano.config.floatX)
    W = np.random.standard_normal((13, 5, 3)).astype(theano.config.floatX)

    X, Z, W = theano_floatx(X, Z, W)

    rnn = SupervisedFastDropoutRnn(4, [10], 3, hidden_transfers=['lstm'],
                                   max_iter=2)
    rnn.mode = 'FAST_COMPILE'
    rnn.fit(X, Z)


@with_setup(*use_test_values('raise'))
def test_srnn_pooling_fit():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((5, 3)).astype(theano.config.floatX)
    W = np.random.standard_normal((5, 3)).astype(theano.config.floatX)

    X, Z, W = theano_floatx(X, Z, W)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], max_iter=2,
                        pooling='sum')
    rnn.fit(X, Z)

    rnn = SupervisedRnn(
        2, [10], 3, hidden_transfers=['tanh'], max_iter=2, imp_weight=True,
        pooling='sum')
    rnn.fit(X, Z, W)


@with_setup(*use_test_values('raise'))
def test_srnn_iter_fit():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    X, Z = theano_floatx(X, Z)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], max_iter=2)
    for i, info in enumerate(rnn.iter_fit(X, Z)):
        if i >= 10:
            break

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], max_iter=2)
    for i, info in enumerate(rnn.iter_fit(X, Z)):
        if i >= 10:
            break


@with_setup(*use_test_values('raise'))
def test_srnn_predict():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], max_iter=2)
    rnn.predict(X)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], max_iter=2)
    rnn.predict(X)


@with_setup(*use_test_values('raise'))
def test_fd_srnn_compile():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    W = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    X, Z, W = theano_floatx(X, Z, W)
    rnn = SupervisedFastDropoutRnn(2, [10], 3, hidden_transfers=['rectifier'],
                                   max_iter=10)

    f_loss, f_dloss = rnn._make_loss_functions()
    f_loss(rnn.parameters.data, X, Z)
    f_dloss(rnn.parameters.data, X, Z)


def test_fd_srnn_fit():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    W = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    X, Z, W = theano_floatx(X, Z, W)

    rnn = SupervisedFastDropoutRnn(2, [10], 3, hidden_transfers=['rectifier'],
                                   max_iter=10)
    rnn.fit(X, Z)

    rnn = SupervisedFastDropoutRnn(
        2, [10, 20], 3, hidden_transfers=['rectifier', 'tanh'], max_iter=2)
    rnn.fit(X, Z)

    rnn = SupervisedFastDropoutRnn(
        2, [10, 20], 3, hidden_transfers=['rectifier', 'tanh'],
        max_iter=2, imp_weight=True)
    rnn.fit(X, Z, W)


def test_fd_srnn_iter_fit():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    X, Z = theano_floatx(X, Z)
    rnn = SupervisedFastDropoutRnn(2, [10], 3, hidden_transfers=['rectifier'],
                                   max_iter=10)
    for i, info in enumerate(rnn.iter_fit(X, Z)):
        if i >= 10:
            break

    rnn = SupervisedFastDropoutRnn(
        2, [10, 20], 3, hidden_transfers=['rectifier', 'tanh'],
        max_iter=2)
    for i, info in enumerate(rnn.iter_fit(X, Z)):
        if i >= 10:
            break


def test_fd_srnn_predict():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)
    rnn = SupervisedFastDropoutRnn(2, [10], 3, hidden_transfers=['rectifier'],
                                   max_iter=10)
    Y = rnn.predict(X)
    assert Y.shape[2] == 6

    rnn = SupervisedFastDropoutRnn(
        2, [10, 20], 3, hidden_transfers=['rectifier', 'tanh'],
        max_iter=2)
    Y = rnn.predict(X)
    assert Y.shape[2] == 6


def test_gn_product_rnn():
    raise SkipTest()
    np.random.seed(1010)
    n_timesteps = 3
    n_inpt = 3
    n_output = 2

    rnn = SupervisedRnn(n_inpt, [1], n_output, out_transfer='sigmoid',
                        loss='squared')
    rnn.parameters.data[:] = np.random.normal(0, 1, rnn.parameters.data.shape)
    X = np.random.random((n_timesteps, 1, n_inpt)).astype(theano.config.floatX)
    Z = np.random.random((n_timesteps, 1, n_output)
                         ).astype(theano.config.floatX)
    X, Z = theano_floatx(X, Z)

    # Calculate the GN explicitly.

    # Shortcuts.
    loss = rnn.exprs['loss']
    output_in = rnn.exprs['output_in']
    p = T.vector('some-vector')

    J = jacobian(output_in[:, 0, :].flatten(), rnn.parameters.flat)

    little_J = T.grad(loss, output_in)[:, 0, :]
    little_H = [[T.grad(little_J[i, j], output_in)
                 for j in range(n_output)]
                for i in range(n_timesteps)]

    f_J = rnn.function(['inpt'], J)
    f_H = rnn.function(['inpt', 'target'], little_H)

    J_ = f_J(X)
    H_ = np.array(f_H(X, Z))[:, :, :, 0, :]
    H_.shape = H_.shape[0] * H_.shape[1], H_.shape[2] * H_.shape[3]

    G_expl = np.dot(J_.T, np.dot(H_, J_))

    p = np.random.random(rnn.parameters.data.shape)
    Gp_expl = np.dot(G_expl, p)

    Hp = rnn._gauss_newton_product()
    args = list(rnn.data_arguments)
    f_Hp = rnn.function(
        ['some-vector'] + args, Hp, explicit_pars=True)
    Gp = f_Hp(rnn.parameters.data, p, X, Z)

    assert np.allclose(Gp, Gp_expl)


@with_setup(*use_test_values('ignore'))
def test_fdrnn_initialize_stds():
    m = SupervisedFastDropoutRnn(
        50, [50], 50,
        ['identity'], 'identity', 'squared')

    inits = dict(par_std=1, par_std_affine=2, par_std_rec=3, par_std_in=4,
                 sparsify_affine=None, sparsify_rec=None, spectral_radius=None)

    p = m.parameters
    p.data[:] = float('nan')

    m.initialize(**inits)

    assert np.isfinite(p.data).all()

    tolerance = 1e-1

    def works(key, tensor):
        std = p[tensor].std()
        target = inits[key]
        success = target - tolerance < std < target + tolerance
        print '%g %g' % (std, target)
        assert success, '%s did not work: %g instead of %g' % (
            key, std, target)

    works('par_std_in', m.rnn.affine_layers[0].weights)
    for l in m.rnn.affine_layers[1:]:
        works('par_std_affine', l.weights)

    for l in m.rnn.recurrent_layers:
        works('par_std_rec', l.weights)


@with_setup(*use_test_values('ignore'))
def test_fdrnn_initialize_sparsify():
    m = SupervisedFastDropoutRnn(
        50, [50], 50,
        ['identity'], 'identity', 'squared')

    inits = dict(par_std=1, sparsify_affine=20, sparsify_rec=35)

    p = m.parameters
    p.data[:] = float('nan')
    m.initialize(**inits)
    assert np.isfinite(p.data).all()

    aff_layers = [i for i in m.rnn.layers if isinstance(i, AffineNonlinear)]
    rec_layers = [i for i in m.rnn.layers if isinstance(i, FDRecurrent)]

    for l in aff_layers:
        w = p[l.weights]
        cond = ((w != 0).sum(axis=0) == inits['sparsify_affine']).all()
        assert cond, 'sparsify affine did not work for %s' % l

    for l in rec_layers:
        w = p[l.weights]
        cond = ((w != 0).sum(axis=0) == inits['sparsify_rec']).all()
        assert cond, 'sparsify recurrent did not work for %s' % l


@with_setup(*use_test_values('ignore'))
def test_fdrnn_initialize_spectral_radius():
    m = SupervisedFastDropoutRnn(
        50, [50], 50,
        ['identity'], 'identity', 'squared')

    inits = dict(par_std=1, spectral_radius=2.5)

    p = m.parameters
    p.data[:] = float('nan')
    m.initialize(**inits)
    assert np.isfinite(p.data).all()

    rec_layers = [i for i in m.rnn.layers if isinstance(i, FDRecurrent)]

    tol = .3

    for l in rec_layers:
        val, vec = np.linalg.eig(p[l.weights])
        sr = abs(sorted(val)[0])
        print abs(sr)
        isr = inits['spectral_radius']
        cond = isr - tol < sr < isr + tol
        assert cond, 'spectral radius in it did not work for %s: %g' % (
            l, sr)


def test_fdrnn_pickle():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    W = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    X, Z, W = theano_floatx(X, Z, W)

    rnn = SupervisedFastDropoutRnn(2, [10], 3, hidden_transfers=['rectifier'],
                                   max_iter=2)
    rnn.fit(X, Z)
    rnn.predict(X)

    pickle.dumps(rnn)
