# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from breze.model.linear import Linear
from breze.model.neural import MultiLayerPerceptron, TwoLayerPerceptron
from breze.model.feature import (
    AutoEncoder, ContractiveAutoEncoder, SparseAutoEncoder, SparseFiltering,
    Rica, DenoisingAutoEncoder, RestrictedBoltzmannMachine)
from breze.model.rim import Rim
from breze.model.sequential import (
    LinearDynamicalSystem,
    SupervisedRecurrentNetwork, UnsupervisedRecurrentNetwork,
    SupervisedLstmRecurrentNetwork, UnsupervisedLstmRecurrentNetwork)

from tools import roughly

from nose.plugins.skip import SkipTest


def test_linear():
    l = Linear(2, 3, 'softabs', 'squared')
    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)), np.random.random((10, 3)))
    fprime(np.random.random((10, 2)), np.random.random((10, 3)))


def test_2lp():
    l = TwoLayerPerceptron(2, 10, 3, 'tanh', 'softabs', 'squared')
    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)), np.random.random((10, 3)))
    fprime(np.random.random((10, 2)), np.random.random((10, 3)))


def test_mlp1():
    l = MultiLayerPerceptron(2, [10], 3, ['tanh'], 'softabs', 'squared')
    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)), np.random.random((10, 3)))
    fprime(np.random.random((10, 2)), np.random.random((10, 3)))


def test_mlp2():
    l = MultiLayerPerceptron(2, [10, 12], 3, ['tanh', 'sigmoid'], 'softabs', 'squared')
    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)), np.random.random((10, 3)))
    fprime(np.random.random((10, 2)), np.random.random((10, 3)))


def test_autoencoder():
    l = AutoEncoder(2, 10, 'tanh', 'identity', 'nces')
    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)))
    fprime(np.random.random((10, 2)))


def test_sae():
    l = SparseAutoEncoder(
        2, 10, 'sigmoid', 'sigmoid', 'nces',
        c_sparsity=5, sparsity_loss='bern_bern_kl',
        sparsity_target=0.05)
    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)))
    fprime(np.random.random((10, 2)))


def test_cae():
    l = ContractiveAutoEncoder(
        2, 10, 'sigmoid', 'sigmoid', 'nces',
        c_jacobian=1.5)
    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)))
    fprime(np.random.random((10, 2)))


def test_sparsefiltering():
    l = SparseFiltering(2, 10, 'softabs')
    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)))
    fprime(np.random.random((10, 2)))


def test_rica():
    l = Rica(2, 10, 'identity', 'softabs', 'identity', 'squared', 1.5)
    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)))
    fprime(np.random.random((10, 2)))


def test_dnae():
    l = DenoisingAutoEncoder(2, 10, 'tanh', 'identity', 'nces')
    f = l.function(['corrupted', 'inpt'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['corrupted', 'inpt'], grad, mode='FAST_COMPILE')

    mtx = np.asarray(np.random.random((10, 2)), dtype=theano.config.floatX)
    f(mtx, mtx)
    fprime(mtx, mtx)


def test_rim():
    l = Rim(2, 3, 1e-4)
    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    d_loss_wrt_pars = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], d_loss_wrt_pars, mode='FAST_COMPILE')

    f(np.asarray(np.random.random((10, 2)), dtype=theano.config.floatX))
    fprime(np.asarray(np.random.random((10, 2)), dtype=theano.config.floatX))


def test_lds():
    l = LinearDynamicalSystem(2, 3)
    l.parameters.data[:] = np.random.random(l.parameters.data.shape)
    nll = -l.exprs['log_likelihood'].mean()
    f = l.function(['inpt'], nll, mode='FAST_COMPILE')

    f(np.random.random((10, 1, 2)))

    # this makes theano crash.
    # grad = T.grad(nll, l.parameters.flat)
    # fprime(np.random.random((10, 2, 2)))


def test_rbm():
    m = RestrictedBoltzmannMachine(2, 3)
    m.parameters.data[:] = np.random.random(m.parameters.data.shape)

    for name, expr in m.exprs.items():
        m.function(
            [m.exprs['inpt'], m.exprs['feature'], m.exprs['n_gibbs_steps']],
            expr,
            on_unused_input='ignore')


def test_lds_values():
    # The following test values were generated with David Barber's LDS
    # implementation coming along with his book 'Bayesian Reasoning and
    # machine learning'.

    l = LinearDynamicalSystem(2, 3)

    l.parameters['transition'] = np.arange(1, 10).reshape((3, 3)).T
    l.parameters['emission'] = np.arange(1, 7).reshape((3, 2))

    l.parameters['visible_noise_mean'] = np.array((-1, 1))
    cov_block = np.arange(1, 5).reshape((2, 2))
    l.parameters['visible_noise_cov'] = (
        np.dot(cov_block, cov_block.T) * 0.2 + np.eye(2) * 100)

    l.parameters['hidden_noise_mean'] = np.array((1, 2, 3))
    cov_block = np.arange(1, 10).reshape((3, 3))
    l.parameters['hidden_noise_cov'] = (
        np.dot(cov_block, cov_block.T) + np.eye(3) * 10)

    l.parameters['hidden_mean_initial'] = -np.array((.1, .2, .3))
    l.parameters['hidden_cov_initial'] = (
        np.dot(cov_block, cov_block.T) * 0.2 + np.eye(3) * 10)

    f_forward = l.function(
        ['inpt'],
        ['filtered_means', 'filtered_covs', 'log_likelihood'],
        mode='FAST_COMPILE')
    f_backward = l.function(
        ['filtered_means', 'filtered_covs'],
        ['smoothed_means', 'smoothed_covs'],
        mode='FAST_COMPILE')

    X = np.array([
        [1, 1.5],
        [2, 2.5],
        [3, 3.5],
        [4, 4.5]])
    X.shape = 4, 1, 2

    f, F, ll = f_forward(X)

    f_desired = np.array([
        [[-0.0395, 0.0649, 0.1694]],
        [[0.5087, 0.2911, 0.0734]],
        [[0.5593, 0.3754, 0.1914]],
        [[0.8622, 0.5272, 0.1921]]])

    F_desired = np.zeros((4, 1, 3, 3))

    F_desired[0, 0] = [
        [9.4992, -1.0806, -1.6604],
        [-1.0806, 7.5570, -3.8054],
        [-1.6604, -3.8054, 4.0497],
    ]

    F_desired[1, 0] = [
        [16.0466, 0.6834, -4.6797],
        [0.6834, 8.0361, -4.6112],
        [-4.6797, -4.6112, 5.4574],
    ]

    F_desired[2, 0] = [
        [18.8713, 1.4414, -5.9885],
        [1.4414, 8.2395, -4.9624],
        [-5.9885, -4.9624, 6.0637],
    ]

    F_desired[3, 0] = [
        [19.7744, 1.6837, -6.4070],
        [1.6837, 8.3045, -5.0747],
        [-6.4070, -5.0747, 6.2577],
    ]

    assert roughly(f, f_desired, 1E-4), 'filtered means not correct'
    assert roughly(F, F_desired, 1E-4), 'filtered covs not correct'
    assert roughly(ll, [-38.3636], 1E-4), 'log likelihood calculated wrong: %f' % ll

    s, S = f_backward(f, F)

    s_desired = np.array([
        [-0.2173, 0.1706, -0.5076, 0.8622],
        [-0.0597, 0.0528, -0.0275, 0.5272],
        [0.0978, -0.0651, 0.4526, 0.1921]]).T

    S_desired = np.zeros((4, 1, 3, 3))
    S_desired[0, 0] = np.array([[5.5498, -2.5166, -0.583],
                                [-2.5166, 6.9683, -3.5468],
                                [-0.583, -3.5468, 3.4894]])

    S_desired[1, 0] = np.array([[7.3348, -2.1811, -1.697],
                                [-2.1811, 7.0324, -3.7542],
                                [-1.697, -3.7542, 4.1886]])

    S_desired[2, 0] = np.array([[9.8391, -1.7404, -3.3199],
                                [-1.7404, 7.11, -4.0396],
                                [-3.3199, -4.0396, 5.2407]])

    S_desired[3, 0] = np.array([[19.7744, 1.6837, -6.407],
                                [1.6837, 8.3045, -5.0747],
                                [-6.407, -5.0747, 6.2577]])

    assert roughly(s[:, 0, :], s_desired, 1E-4), 'smooooothed means not correct'
    assert roughly(S, S_desired, 1E-4), 'smooooothed covs not correct'


def test_lds_shapes():
    raise SkipTest()
    n_inpt = 2
    n_hidden = 3

    inpt = T.tensor3('inpt')
    inpt.tag.test_value = np.random.random((10, 2, n_inpt))

    transition = T.matrix('transitions')
    transition.tag.test_value = np.random.random((n_hidden, n_hidden))

    emission = T.matrix('emissions')
    emission.tag.test_value = np.random.random((n_hidden, n_inpt))

    visible_noise_mean = T.vector('visible_noise_mean')
    visible_noise_mean.tag.test_value = np.random.random(n_inpt)

    visible_noise_cov = T.matrix('visible_noise_cov')
    visible_noise_cov.tag.test_value = np.random.random((n_inpt, n_inpt))

    hidden_noise_mean = T.vector('hidden_noise_mean')
    hidden_noise_mean.tag.test_value = np.random.random(n_hidden)

    hidden_noise_cov = T.matrix('hidden_noise_cov')
    hidden_noise_cov.tag.test_value = np.random.random((n_hidden, n_hidden))

    hidden_mean_initial = T.vector('hidden_mean_initial')
    hidden_mean_initial.tag.test_value = np.random.random(n_hidden)

    hidden_cov_initial = T.matrix('hidden_cov_initial')
    hidden_cov_initial.tag.test_value = np.random.random((n_hidden, n_hidden))

    LinearDynamicalSystem.make_exprs(
        inpt, transition, emission,
        visible_noise_mean, visible_noise_cov,
        hidden_noise_mean, hidden_noise_cov,
        hidden_mean_initial, hidden_cov_initial)
    theano.config.compute_test_value = 'off'


def test_srnn():
    l = SupervisedRecurrentNetwork(2, 3, 1, 'sigmoid', 'identity', 'squared')
    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    d_loss_wrt_pars = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], d_loss_wrt_pars,
                        mode='FAST_COMPILE')

    X = np.random.random((10, 3, 2)).astype(theano.config.floatX)
    Z = np.random.random((10, 3, 1)).astype(theano.config.floatX)

    f(X, Z)
    fprime(X, Z)


def test_dsrnn():
    l = SupervisedRecurrentNetwork(2, [3] * 2, 1, ['sigmoid'] * 2, 'identity', 'squared')
    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    d_loss_wrt_pars = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], d_loss_wrt_pars,
                        mode='FAST_COMPILE')

    X = np.random.random((10, 3, 2)).astype(theano.config.floatX)
    Z = np.random.random((10, 3, 1)).astype(theano.config.floatX)

    f(X, Z)
    fprime(X, Z)


def test_usrnn():
    l = UnsupervisedRecurrentNetwork(
        2, 3, 1, 'sigmoid', 'identity', lambda x: abs(x))
    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    d_loss_wrt_pars = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], d_loss_wrt_pars,
                        mode='FAST_COMPILE')

    X = np.random.random((10, 3, 2)).astype(theano.config.floatX)

    f(X)
    fprime(X)


def test_pooling_rnn():
    l = SupervisedRecurrentNetwork(2, 3, 1, 'sigmoid', 'identity', 'ncac', 'mean')
    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    d_loss_wrt_pars = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], d_loss_wrt_pars,
                        mode='FAST_COMPILE')

    X = np.random.random((10, 30, 2)).astype(theano.config.floatX)
    Z = np.random.random((30, 1)).astype(theano.config.floatX)

    f(X, Z)
    fprime(X, Z)


def test_slstmrnn():
    l = SupervisedLstmRecurrentNetwork(
        2, 5, 1, 'sigmoid', 'identity', 'squared')

    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    d_loss_wrt_pars = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], d_loss_wrt_pars,
                        mode='FAST_COMPILE')

    X = np.random.random((10, 3, 2)).astype(theano.config.floatX)
    Z = np.random.random((10, 3, 1)).astype(theano.config.floatX)

    f(X, Z)
    fprime(X, Z)


def test_pooling_slstmrnn():
    l = SupervisedLstmRecurrentNetwork(
        2, 3, 1, 'sigmoid', 'identity', 'ncac', 'mean')
    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    d_loss_wrt_pars = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], d_loss_wrt_pars,
                        mode='FAST_COMPILE')

    X = np.random.random((10, 30, 2)).astype(theano.config.floatX)
    Z = np.random.random((30, 1)).astype(theano.config.floatX)

    f(X, Z)
    fprime(X, Z)


def test_uslstmrnn():
    l = UnsupervisedLstmRecurrentNetwork(
        2, 5, 1, 'sigmoid', 'identity', lambda X: abs(X))

    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    d_loss_wrt_pars = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], d_loss_wrt_pars,
                        mode='FAST_COMPILE')

    X = np.random.random((10, 3, 2)).astype(theano.config.floatX)

    f(X)
    fprime(X)


def test_pooling_uslstmrnn():
    l = UnsupervisedLstmRecurrentNetwork(
        2, 3, 1, 'sigmoid', 'identity', lambda X: abs(X), 'mean')

    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    d_loss_wrt_pars = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], d_loss_wrt_pars,
                        mode='FAST_COMPILE')

    X = np.random.random((10, 30, 2)).astype(theano.config.floatX)

    f(X)
    fprime(X)
