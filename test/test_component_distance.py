# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from breze.component.distance import (
    absolute, squared, bernoulli_cross_entropy, cross_entropy,
    distance_matrix, nominal_cross_entropy, nca)
from tools import roughly


test_X = np.array([
        [.1, .2, .3],
        [.2, .1, .2]], dtype=theano.config.floatX)


test_Y = np.array([
        [.13, .21, .32],
        [.23, .17, .25]], dtype=theano.config.floatX)


test_Xc = np.array([0, 2], dtype=np.int32)


def test_absolute():
    X, Y = T.matrix(), T.matrix()
    dist = absolute(X, Y)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    assert res == 0.21, 'absolute distance not working'


def test_absolute_rowwise():
    X, Y = T.matrix(), T.matrix()
    dist = absolute(X, Y, axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.06, 0.15])
    assert correct, 'absolute distance rowwise not working'


def test_absolute_colwise():
    X, Y = T.matrix(), T.matrix()
    dist = absolute(X, Y, axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.06, 0.08, 0.07])
    assert correct, 'absolute distance colwise not working'


def test_squared():
    X, Y = T.matrix(), T.matrix()
    dist = squared(X, Y)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    assert roughly(res, 0.0097), 'squared distance not working'


def test_squared_rowwise():
    X, Y = T.matrix(), T.matrix()
    dist = squared(X, Y, axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.0014, 0.0083])
    assert correct, 'squared distance rowwise not working'


def test_squared_colwise():
    X, Y = T.matrix(), T.matrix()
    dist = squared(X, Y, axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.0018, 0.005, 0.0029])
    assert correct, 'squared distance colwise not working'


def test_cross_entropy():
    X, Y = T.matrix(), T.matrix()
    dist = cross_entropy(X, Y)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, 1.6063716678910529)
    assert correct, 'cross_entropy distance not working'


def test_cross_entropy_rowwise():
    X, Y = T.matrix(), T.matrix()
    dist = cross_entropy(X, Y, axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.85798192, 0.74838975])
    assert correct, 'bernoulli_cross_entropy distance rowwise not working'


def test_cross_entropy_colwise():
    X, Y = T.matrix(), T.matrix()
    dist = cross_entropy(X, Y, axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [[0.49795728, 0.48932523, 0.61908916]])
    assert correct, 'bernoulli_cross_entropy distance colwise not working'


def test_nominal_cross_entropy():
    Xc, Y = T.ivector(), T.matrix()
    dist = nominal_cross_entropy(Xc, Y)
    f = theano.function([Xc, Y], dist, mode='FAST_COMPILE')
    res = f(test_Xc, test_Y)
    correct = roughly(res, -np.log(test_Y)[np.arange(test_Xc.shape[0]), test_Xc])
    assert correct, 'nominal_cross_entropy distance not working'


def test_distance_matrix():
    X = T.matrix()
    D = distance_matrix(X)
    f = theano.function([X], D, mode='FAST_COMPILE')
    x = np.array([[1], [2], [3]])
    res = f(x)
    print res
    correct = roughly(res, np.array([[0, 1, 4], [1, 0, 1], [4, 1, 0]]))
    assert correct, 'distance matrix not working right'


def test_nca():
    theano.config.compute_test_value = 'off'
    X = T.matrix()
    X.tag.test_value = np.random.random((20, 10))
    Y = T.matrix()
    Y.tag.test_value = np.random.random((20, 1)) > 0.5
    expr = nca(X, Y)
    theano.config.compute_test_value = 'off'
