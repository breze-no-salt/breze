# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from breze.arch.component.loss import absolute, squared, nce, nnce, ncac, drlim1
from tools import roughly

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
    assert res == 0.21, 'absolute loss not working'


@with_setup(test_values_raise, test_values_off)
def test_absolute_rowwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = absolute(X, Y).sum(axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.06, 0.15])
    assert correct, 'absolute loss rowwise not working'


@with_setup(test_values_raise, test_values_off)
def test_absolute_colwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = absolute(X, Y).sum(axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.06, 0.08, 0.07])
    assert correct, 'absolute loss colwise not working'


@with_setup(test_values_raise, test_values_off)
def test_squared():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = squared(X, Y).sum()
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    assert roughly(res, 0.0097), 'squared loss not working'


@with_setup(test_values_raise, test_values_off)
def test_squared_rowwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = squared(X, Y).sum(axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.0014, 0.0083])
    assert correct, 'squared loss rowwise not working'


@with_setup(test_values_raise, test_values_off)
def test_squared_colwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X
    Y.tag.test_value = test_Y
    dist = squared(X, Y).sum(axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.0018, 0.005, 0.0029])
    assert correct, 'squared loss colwise not working'


@with_setup(test_values_raise, test_values_off)
def test_neg_cross_entropy():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X > 0.2
    Y.tag.test_value = test_Y
    dist = nce(X, Y).sum()
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, 1.6063716678910529)
    assert correct, 'nce loss not working'


@with_setup(test_values_raise, test_values_off)
def test_neg_cross_entropy_rowwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X > 0.2
    Y.tag.test_value = test_Y
    dist = nce(X, Y).sum(axis=1)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [0.85798192, 0.74838975])
    assert correct, 'nce loss rowwise not working'


@with_setup(test_values_raise, test_values_off)
def test_neg_cross_entropy_colwise():
    X, Y = T.matrix(), T.matrix()
    X.tag.test_value = test_X > 0.2
    Y.tag.test_value = test_Y
    dist = nce(X, Y).sum(axis=0)
    f = theano.function([X, Y], dist, mode='FAST_COMPILE')
    res = f(test_X, test_Y)
    correct = roughly(res, [[0.49795728, 0.48932523, 0.61908916]])
    assert correct, 'nce loss colwise not working'


@with_setup(test_values_raise, test_values_off)
def test_nominal_neg_cross_entropy():
    Xc, Y = T.ivector(), T.matrix()
    Xc.tag.test_value = (test_X > 0.2).argmax(axis=1).astype('uint8')
    Y.tag.test_value = test_Y
    dist = nnce(Xc, Y).sum()
    f = theano.function([Xc, Y], dist, mode='FAST_COMPILE')
    res = f(test_Xc, test_Y)
    desired = -np.log(test_Y)[np.arange(test_Xc.shape[0]), test_Xc].sum()
    correct = roughly(res, desired)
    print res
    print desired
    assert correct, 'nnce loss not working'


@with_setup(test_values_raise, test_values_off)
def test_nca():
    X = T.matrix()
    X.tag.test_value = np.random.random((20, 10))
    Y = T.matrix()
    Y.tag.test_value = np.random.random((20, 1)) > 0.5
    expr = ncac(X, Y)


@with_setup(test_values_raise, test_values_off)
def test_drlim():
    X = T.matrix()
    X.tag.test_value = np.random.random((20, 10))
    Y = T.matrix()
    Y.tag.test_value = np.random.random((10, 1)) > 0.5
    expr = drlim1(Y, X)
