# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from breze.component.distance import manhattan, euclidean, bernoulli_cross_entropy
from tools import roughly


test_X = np.array([
        [.1, .2, .3],
        [.2, .1, .2]])


test_Y = np.array([
        [.13, .21, .32],
        [.23, .17, .25]])


def test_manhattan():
    X, Y = T.matrix(), T.matrix()
    dist = manhattan(X, Y)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    assert res == 0.21, 'manhattan distance not working'


def test_manhattan_rowwise():
    X, Y = T.matrix(), T.matrix()
    dist = manhattan(X, Y, axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.06, 0.15])
    assert correct, 'manhattan distance rowwise not working'


def test_manhattan_colwise():
    X, Y = T.matrix(), T.matrix()
    dist = manhattan(X, Y, axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.06, 0.08, 0.07])
    assert correct, 'manhattan distance colwise not working'


def test_euclidean():
    X, Y = T.matrix(), T.matrix()
    dist = euclidean(X, Y)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    assert roughly(res, 0.0097), 'euclidean distance not working'


def test_euclidean_rowwise():
    X, Y = T.matrix(), T.matrix()
    dist = euclidean(X, Y, axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.0014, 0.0083])
    assert correct, 'euclidean distance rowwise not working'


def test_euclidean_colwise():
    X, Y = T.matrix(), T.matrix()
    dist = euclidean(X, Y, axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.0018, 0.005, 0.0029])
    assert correct, 'euclidean distance colwise not working'


def test_bernoulli_cross_entropy():
    X, Y = T.matrix(), T.matrix()
    dist = bernoulli_cross_entropy(X, Y)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, 1.6063716678910529)
    assert correct, 'bernoulli_cross_entropy distance not working'


def test_bernoulli_cross_entropy_rowwise():
    X, Y = T.matrix(), T.matrix()
    dist = bernoulli_cross_entropy(X, Y, axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.85798192, 0.74838975])
    assert correct, 'bernoulli_cross_entropy distance rowwise not working'


def test_bernoulli_cross_entropy_colwise():
    X, Y = T.matrix(), T.matrix()
    dist = bernoulli_cross_entropy(X, Y, axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [[0.49795728, 0.48932523, 0.61908916]])
    assert correct, 'bernoulli_cross_entropy distance colwise not working'
