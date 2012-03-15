# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from breze.component.transfer import (
    sigmoid, tanh, tanhplus, rectified_linear, soft_rectified_linear,
    logproduct_of_t)

from tools import roughly 


test_matrix = np.array([
        [-2, 3.2, 4.5, -100.2],
        [1, -1, 2, 0]])


def test_tanh():
    inpt = T.matrix()
    expr = tanh(inpt)
    f = theano.function([inpt], expr)
    result = f(test_matrix)
    desired = np.array([
        [-0.96402758, 0.9966824, 0.99975321, -1.],
        [0.76159416, -0.76159416, 0.96402758, 0.]])
    correct = roughly(result, desired)
    assert correct, 'tanh not working'



def test_tanhplus():
    inpt = T.matrix()
    expr = tanhplus(inpt)
    f = theano.function([inpt], expr)
    result = f(test_matrix)
    desired = np.array([
        [-2.96402758, 4.1966824 , 5.49975321, -101.2],
        [ 1.76159416,-1.76159416, 2.96402758, 0.]])
    correct = roughly(result, desired)
    assert correct, 'tanh plus not working'


def test_sigmoid():
    inpt = T.matrix()
    expr = sigmoid(inpt)
    f = theano.function([inpt], expr)
    result = f(test_matrix)
    desired = np.array([
        [1.19202922e-01, 9.60834277e-01, 9.89013057e-01, 3.04574061e-44],
        [7.31058579e-01, 2.68941421e-01, 8.80797078e-01, 5.00000000e-01]])

    correct = roughly(result, desired)
    assert correct, 'sigmoid not working'


def test_rectified_linear():
    inpt = T.matrix()
    expr = rectified_linear(inpt)
    f = theano.function([inpt], expr)
    result = f(test_matrix)
    desired = np.array([
       [0., 3.2, 4.5, 0.],
       [1., 0., 2., 0.]])

    correct = roughly(result, desired)
    assert correct, 'relu not working'


def test_soft_rectified_linear():
    inpt = T.matrix()
    expr = soft_rectified_linear(inpt)
    f = theano.function([inpt], expr)
    result = f(test_matrix)
    desired = np.array([
       [0.12692801, 3.23995333, 4.51104774,  0.],
       [1.31326169, 0.31326169, 2.12692801,  0.69314718]])

    correct = roughly(result, desired)
    assert correct, 'soft relu not working'


def test_log_product_of_t():
    inpt = T.matrix()
    expr = logproduct_of_t(inpt)
    f = theano.function([inpt], expr)
    result = f(test_matrix)
    desired = np.array([
       [1.60943791, 2.41947884, 3.0563569 , 9.21443597],
       [0.69314718, 0.69314718, 1.60943791, 0.]])

    correct = roughly(result, desired)
    assert correct, 'pot not working'
