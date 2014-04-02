from math import sqrt

import numpy as np

import theano
from theano.tensor import tensor4

from breze.arch.model.neural.cnn import pad, perform_pooling, perform_lrnorm


def prepare_array(arr):
    arr = np.array(arr)
    return arr.reshape((1, 1) + arr.shape)


def test_pad():
    inpt = prepare_array([[1, 1], [1, 1]])
    output = prepare_array([[0, 0, 0, 0], [0, 1, 1, 0],
                            [0, 1, 1, 0], [0, 0, 0, 0]])
    inpt_expr = tensor4('input')
    output_expr = pad(inpt_expr, 1)
    f = theano.function([inpt_expr], output_expr)
    assert np.allclose(f(inpt), output)


def test_pooling():
    shift = [[0, 1], [0, 1]]
    pool_shape = [2, 2]
    limits = [2, 2]
    inpt = prepare_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = prepare_array([[5, 6], [8, 9]])
    inpt_expr = tensor4('input')
    output_expr = perform_pooling(inpt_expr, shift, pool_shape, limits)
    f = theano.function([inpt_expr], output_expr)
    assert np.allclose(f(inpt), output)
    shift = [[0], [0, 1]]
    pool_shape = [2, 2]
    limits = [1, 2]
    output = prepare_array([[5, 6]])
    inpt_expr = tensor4('input')
    output_expr = perform_pooling(inpt_expr, shift, pool_shape, limits)
    f = theano.function([inpt_expr], output_expr)
    assert np.allclose(f(inpt), output)
    shift = [[0, 1], [0, 1]]
    pool_shape = [1, 2]
    limits = [3, 2]
    output = prepare_array([[2, 3], [5, 6], [8, 9]])
    inpt_expr = tensor4('input')
    output_expr = perform_pooling(inpt_expr, shift, pool_shape, limits)
    f = theano.function([inpt_expr], output_expr)
    assert np.allclose(f(inpt), output)


def test_lrnorm():
    alpha = 2
    beta = 0.5
    N = 2
    inpt = prepare_array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
    output = prepare_array([[2./3., 2./sqrt(13.), 2./3.],
                            [2./sqrt(13.), 2./sqrt(19.), 2./sqrt(13.)],
                            [2./3., 2./sqrt(13.), 2./3.]])
    inpt_expr = tensor4('input')
    output_expr = perform_lrnorm(inpt_expr, [alpha, beta, N])
    f = theano.function([inpt_expr], output_expr)
    assert np.allclose(f(inpt), output)


