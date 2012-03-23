# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from breze.model.linear import Linear
from breze.model.neural import MultiLayerPerceptron, TwoLayerPerceptron
from breze.model.feature import (
    AutoEncoder, ContractiveAutoEncoder, SparseAutoEncoder, SparseFiltering,
    Rica)


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


def test_mlp():
    l = MultiLayerPerceptron(2, [10], 3, ['tanh'], 'softabs', 'squared')
    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)), np.random.random((10, 3)))
    fprime(np.random.random((10, 2)), np.random.random((10, 3)))


def test_mlp():
    l = MultiLayerPerceptron(2, [10, 12], 3, ['tanh', 'sigmoid'], 'softabs', 'squared')
    f = l.function(['inpt', 'target'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt', 'target'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)), np.random.random((10, 3)))
    fprime(np.random.random((10, 2)), np.random.random((10, 3)))


def test_autoencoder():
    l = AutoEncoder(2, 10, 'tanh', 'identity', 'bernoulli_cross_entropy')
    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)))
    fprime(np.random.random((10, 2)))


def test_sae():
    l = SparseAutoEncoder(
        2, 10, 'sigmoid', 'sigmoid', 'bernoulli_cross_entropy',
        c_sparsity=5, sparsity_loss='bernoulli_cross_entropy', 
        sparsity_target=0.05)
    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)))
    fprime(np.random.random((10, 2)))


def test_cae():
    l = ContractiveAutoEncoder(
        2, 10, 'sigmoid', 'sigmoid', 'bernoulli_cross_entropy',
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
    l = Rica(2, 10, 'tanh', 'identity', 'squared', 1.5)
    f = l.function(['inpt'], 'loss', mode='FAST_COMPILE')
    grad = T.grad(l.exprs['loss'], l.parameters.flat)
    fprime = l.function(['inpt'], grad, mode='FAST_COMPILE')

    f(np.random.random((10, 2)))
    fprime(np.random.random((10, 2)))
