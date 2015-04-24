# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.gradient import jacobian

from breze.learn.rnn import (
    SupervisedFastDropoutRnn,
    SupervisedRnn, UnsupervisedRnn,
    SupervisedLstmRnn, UnsupervisedLstmRnn)
from breze.learn.utils import theano_floatx

from nose.plugins.skip import SkipTest


def test_srnn_fit():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    W = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)

    X, Z, W = theano_floatx(X, Z, W)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], max_iter=2)
    rnn.fit(X, Z)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'],
        max_iter=2, imp_weight=True)
    rnn.fit(X, Z, W)


def test_srnn_pooling_fit():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((5, 3)).astype(theano.config.floatX)
    W = np.random.standard_normal((5, 3)).astype(theano.config.floatX)

    X, Z, W = theano_floatx(X, Z, W)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], max_iter=2,
                        pooling='sum')
    rnn.fit(X, Z)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'],
        max_iter=2, imp_weight=True, pooling='sum')
    rnn.fit(X, Z, W)


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


def test_srnn_predict():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], max_iter=2)
    rnn.predict(X)

    rnn = SupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], max_iter=2)
    rnn.predict(X)


def test_fd_srnn_fit():
    raise SkipTest()
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    W = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    X, Z, W = theano_floatx(X, Z, W)
    rnn = SupervisedFastDropoutRnn(2, [10], 3, hidden_transfers=['rectifier'], max_iter=10)
    rnn.fit(X, Z)

    rnn = SupervisedFastDropoutRnn(
        2, [10, 20], 3, hidden_transfers=['rectifier', 'tanh'], max_iter=2)
    rnn.fit(X, Z)

    rnn = SupervisedFastDropoutRnn(
        2, [10, 20], 3, hidden_transfers=['rectifier', 'tanh'],
        max_iter=2, imp_weight=True)
    rnn.fit(X, Z, W)


def test_fd_srnn_iter_fit():
    raise SkipTest()
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    X, Z = theano_floatx(X, Z)
    rnn = SupervisedFastDropoutRnn(2, [10], 3, hidden_transfers=['rectifier'], max_iter=10)
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
    raise SkipTest()
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)
    rnn = SupervisedFastDropoutRnn(2, [10], 3, hidden_transfers=['rectifier'], max_iter=10)
    rnn.predict(X)

    rnn = SupervisedFastDropoutRnn(
        2, [10, 20], 3, hidden_transfers=['rectifier', 'tanh'],
        max_iter=2)
    rnn.predict(X)


def test_usrnn_fit():
    raise SkipTest()
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)
    rnn = UnsupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], loss=lambda x: T.log(x), max_iter=10)
    rnn.fit(X)


def test_usrnn_iter_fit():
    raise SkipTest()
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)
    rnn = UnsupervisedRnn(2, [10], 3, hidden_transfers=['tanh'], loss=lambda x: T.log(x), max_iter=10)
    for i, info in enumerate(rnn.iter_fit(X)):
        if i >= 10:
            break


def test_usrnn_transform():
    raise SkipTest()
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)
    rnn = UnsupervisedRnn(2, [10], 3, hidden_transfers=['tanh'],
                          loss=lambda x: T.log(x), max_iter=10)
    rnn.transform(X)


def test_slstm():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    X, Z = theano_floatx(X, Z)
    rnn = SupervisedLstmRnn(2, [10], 3, hidden_transfers=['sigmoid'], max_iter=10)
    rnn.fit(X, Z)


def test_slstm_iter_fit():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    Z = np.random.standard_normal((10, 5, 3)).astype(theano.config.floatX)
    X, Z = theano_floatx(X, Z)
    rnn = SupervisedLstmRnn(2, [10], 3, hidden_transfers=['sigmoid'], max_iter=10)
    for i, info in enumerate(rnn.iter_fit(X, Z)):
        if i >= 10:
            break


def test_slstm_predict():
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)
    rnn = SupervisedLstmRnn(2, [10], 3, hidden_transfers=['sigmoid'], max_iter=10)
    rnn.predict(X)


def test_uslstm_fit():
    raise SkipTest()
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)
    rnn = UnsupervisedLstm(2, [10], 3, loss=lambda x: T.log(x), max_iter=10)
    rnn.fit(X)


def test_uslstm_iter_fit():
    raise SkipTest()
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)
    rnn = UnsupervisedLstm(2, [10], 3, loss=lambda x: T.log(x), max_iter=10)
    for i, info in enumerate(rnn.iter_fit(X)):
        if i >= 10:
            break


def test_uslstm_transform():
    raise SkipTest()
    X = np.random.standard_normal((10, 5, 2)).astype(theano.config.floatX)
    X, = theano_floatx(X)
    rnn = UnsupervisedLstm(2, [10], 3, loss=lambda x: T.log(x), max_iter=10)
    rnn.transform(X)


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
    Z = np.random.random((n_timesteps, 1, n_output)).astype(theano.config.floatX)
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
