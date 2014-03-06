# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from breze.arch.component.loss import (
        absolute, squared, cat_ce, ncat_ce, ncac, drlim1)

from nose.tools import with_setup

from tools import test_values_off, test_values_raise


test_X = np.array([[.1, .2, .3],
                   [.2, .1, .2]],
                  dtype=theano.config.floatX)


test_Y = np.array([[.13, .21, .32],
                   [.23, .17, .25]],
                  dtype=theano.config.floatX)


test_Xc = np.array([0, 2], dtype=np.int32)


@with_setup(test_values_raise, test_values_off)
def test_absolute():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = absolute(X, Y).sum()
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    print res
    assert np.allclose(res, 0.21), 'absolute loss not working'


@with_setup(test_values_raise, test_values_off)
def test_absolute_rowwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = absolute(X, Y).sum(axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = np.allclose(res, [0.06, 0.15])
    assert correct, 'absolute loss rowwise not working'


@with_setup(test_values_raise, test_values_off)
def test_absolute_colwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = absolute(X, Y).sum(axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = np.allclose(res, [0.06, 0.08, 0.07])
    assert correct, 'absolute loss colwise not working'


@with_setup(test_values_raise, test_values_off)
def test_squared():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = squared(X, Y).sum()
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    assert np.allclose(res, 0.0097), 'squared loss not working'


@with_setup(test_values_raise, test_values_off)
def test_squared_rowwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = squared(X, Y).sum(axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = np.allclose(res, [0.0014, 0.0083])
    assert correct, 'squared loss rowwise not working'


@with_setup(test_values_raise, test_values_off)
def test_squared_colwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = squared(X, Y).sum(axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = np.allclose(res, [0.0018, 0.005, 0.0029])
    assert correct, 'squared loss colwise not working'


@with_setup(test_values_raise, test_values_off)
def test_neg_cross_entropy():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X > 0.2
    Y.tag.test_value = test_Y
    dist = cat_ce(X, Y).sum()
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = np.allclose(res, 1.6063716678910529)
    assert correct, 'cat_ce loss not working'


@with_setup(test_values_raise, test_values_off)
def test_neg_cross_entropy_rowwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X > 0.2
    Y.tag.test_value = test_Y
    dist = cat_ce(X, Y).sum(axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = np.allclose(res, [0.85798192, 0.74838975])
    assert correct, 'cat_ce loss rowwise not working'


@with_setup(test_values_raise, test_values_off)
def test_neg_cross_entropy_colwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X > 0.2
    Y.tag.test_value = test_Y
    dist = cat_ce(X, Y).sum(axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = np.allclose(res, [[0.49795728, 0.48932523, 0.61908916]])
    assert correct, 'cat_ce loss colwise not working'


@with_setup(test_values_raise, test_values_off)
def test_nominal_neg_cross_entropy():
    Xc, Y = T.ivector(), T.matrix()
    Xc.tag.test_value = (test_X > 0.2).argmax(axis=1).astype('uint8')
    Y.tag.test_value = test_Y
    dist = ncat_ce(Xc, Y).sum()
    f = theano.function([Xc, Y], dist, mode='FAST_COMPILE')
    res = f(test_Xc, test_Y)
    desired = -np.log(test_Y)[np.arange(test_Xc.shape[0]), test_Xc].sum()
    correct = np.allclose(res, desired)
    print res
    print desired
    assert correct, 'ncat_ce loss not working'


@with_setup(test_values_raise, test_values_off)
def test_nca():
    X = T.matrix()
    X.tag.test_value = np.random.random((20, 10)).astype(theano.config.floatX)
    Y = T.matrix()
    Y.tag.test_value = np.random.random((20, 1)).astype(theano.config.floatX) > 0.5
    expr = ncac(X, Y)


@with_setup(test_values_raise, test_values_off)
def test_drlim():
    X = T.matrix()
    X.tag.test_value = np.random.random((20, 10)).astype(theano.config.floatX)
    Y = T.matrix()
    Y.tag.test_value = np.random.random((10, 1)).astype(theano.config.floatX) > 0.5
    expr = drlim1(Y, X)
