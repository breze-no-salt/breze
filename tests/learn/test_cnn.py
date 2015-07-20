# -*- coding: utf-8 -*-
import numpy as np

from breze.learn.cnn import SimpleCnn2d, Lenet
from breze.learn.utils import theano_floatx
import theano

theano.config.exception_verbosity = 'high'


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

    theano.config.compute_test_value = 'raise'
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

    theano.config.compute_test_value = 'raise'
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
