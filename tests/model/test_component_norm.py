# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from breze.model.component.norm import l1, l2, exp

from tools import roughly


test_arr = np.array([1, -1, 2, 0])
test_matrix = np.array([
        [-2, 3.2, 4.5, -100.2],
        [1, -1, 2, 0]])


def test_l1_vector():
    inpt = T.vector()
    norm = l1(inpt)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    assert f(test_arr) == 4, 'l1 norm for vector not working'


def test_l2_vector():
    inpt = T.vector()
    norm = l2(inpt)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    assert f(test_arr) == 6, 'l2 norm for vector not working'


def test_exp_vector():
    inpt = T.vector()
    norm = exp(inpt)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    assert f(test_arr) == 11.475217368561138, 'exp norm for vector not working'


def test_l1_matrix():
    inpt = T.matrix()
    norm = l1(inpt)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    assert f(test_matrix) == 113.9, 'l1 norm for matrix not working'


def test_l2_matrix():
    inpt = T.matrix()
    norm = l2(inpt)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    assert f(test_matrix) == 10080.53, 'l2 norm for matrix not working'


def test_exp_matrix():
    inpt = T.matrix()
    norm = exp(inpt)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    assert f(test_matrix) == 126.16021414942891, 'exp norm for matrix not working'


def test_l1_matrix_rowwise():
    inpt = T.matrix()
    norm = l1(inpt, axis=1)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    res = f(test_matrix)
    correct = (res == [109.9, 4.]).all()
    assert correct, 'l1 norm rowwise not working'


def test_l2_matrix_rowwise():
    inpt = T.matrix()
    norm = l2(inpt, axis=1)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    res = f(test_matrix)
    correct = (res == [1.00745300e+04, 6.]).all()
    assert correct, 'l2 norm rowwise not working'


def test_exp_matrix_rowwise():
    inpt = T.matrix()
    norm = exp(inpt, axis=1)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    res = f(test_matrix)
    correct = roughly(res, [114.68499678, 11.47521737])
    assert correct, 'exp norm rowwise not working'


def test_l1_matrix_colwise():
    inpt = T.matrix()
    norm = l1(inpt, axis=0)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    res = f(test_matrix)
    correct = roughly(res, [3., 4.2, 6.5, 100.2])
    assert correct, 'l1 norm colwise not working'


def test_l2_matrix_colwise():
    inpt = T.matrix()
    norm = l2(inpt, axis=0)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    res = f(test_matrix)
    correct = roughly(
            res,
            [5., 1.12400000e+01, 2.42500000e+01, 1.00400400e+04])
    assert correct, 'l2 norm colwise not working'


def test_exp_matrix_colwise():
    inpt = T.matrix()
    norm = exp(inpt, axis=0)
    f = theano.function([inpt], norm, mode='FAST_COMPILE')
    res = f(test_matrix)
    correct = roughly(res, [2.85361711, 24.90040964, 97.4061874, 1.])
    assert correct, 'exp norm colwise not working'
