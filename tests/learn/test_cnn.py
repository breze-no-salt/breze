# -*- coding: utf-8 -*-


import numpy as np
import theano

from nose.tools import with_setup

from breze.learn.cnn import SimpleCnn2d, Lenet
from breze.learn.utils import theano_floatx
from breze.utils.testhelpers import use_test_values


@with_setup(*use_test_values('raise'))
def test_simplecnn2d_fit():
    image_height, image_width = 4, 4
    X = np.random.standard_normal((11, 1, image_height, image_width))
    Z = np.random.random((11, 1)) > 0.5
    X, Z = theano_floatx(X, Z)

    n_hiddens = [5, 2]
    transfers = ['tanh', 'tanh']
    filter_shapes = [(2, 2), (2, 2)]
    n_channel = 1
    n_output = 1
    out_transfer = 'identity'
    loss = 'squared'

    m = SimpleCnn2d(
        image_height, image_width, n_channel,
        n_hiddens,
        filter_shapes,
        n_output,
        transfers, out_transfer,
        loss)

    f_predict = m.function(['inpt'], 'output', mode='FAST_COMPILE')
    f_predict(X)

    m.fit(X, Z)


@with_setup(*use_test_values('raise'))
def test_lenet():
    image_height, image_width = 16, 16
    X = np.random.standard_normal((11, 1, image_height, image_width))
    Z = np.random.random((11, 1)) > 0.5
    X, Z = theano_floatx(X, Z)

    n_hiddens_conv = [5, 2]
    filter_shapes = [(2, 2), (2, 2)]
    pool_shapes = [(2, 2), (2, 2)]
    n_hiddens_full = [20]
    transfers_conv = ['tanh', 'tanh']
    transfers_full = ['rectifier']
    n_channel = 1
    n_output = 1
    out_transfer = 'identity'
    loss = 'squared'

    m = Lenet(
        image_height, image_width, n_channel,
        n_hiddens_conv,
        filter_shapes,
        pool_shapes,
        n_hiddens_full,
        n_output,
        transfers_conv, transfers_full, out_transfer,
        loss)

    f_predict = m.function(['inpt'], 'output', mode='FAST_COMPILE')
    f_predict(X)

    m.fit(X, Z)
