# -*- coding: utf-8 -*-


import numpy as np

import theano
import theano.tensor as T
from breze.arch.construct import base
from breze.arch.construct.layer import simple, sequential


X = np.zeros((10, 2))
Z = np.zeros((10, 3))

theano.config.compute_test_value = 'raise'


def test_simple_stack():
    inpt = T.matrix('inpt')
    target = T.matrix('target')

    layers = [
        simple.AffineNonlinear(2, 2),
        simple.AffineNonlinear(2, 2),
        simple.AffineNonlinear(2, 3),
    ]
    loss = simple.SupervisedLoss('squared', target=target)

    s = base.SupervisedStack(layers=layers, loss=loss)

    s.forward(inpt)

    s.function(['inpt', 'target'], 'loss')
    Y = s.predict(X)
    assert Y.shape == (10, 3), 'shape of output not right'


def test_sequential():
    inpt = T.tensor3('inpt')
    inpt.tag.test_value = np.zeros((2, 3, 4))
    seq_to_static = sequential.SequentialToStatic()
    output = seq_to_static(inpt)
    print output
    recons = seq_to_static.inverse(*output)

    assert output[0].tag.test_value.shape == (6, 4)
    assert recons[0].tag.test_value.shape == (2, 3, 4)


def test_rnn():
    inpt = T.tensor3('inpt')
    inpt.tag.test_value = np.empty((10, 5, 2))

    target = T.tensor3('target')
    target.tag.test_value = np.empty((10, 5, 3))

    seq_to_static = sequential.SequentialToStatic()

    layers = [
        seq_to_static,
        simple.AffineNonlinear(2, 2),
        seq_to_static.inverse,
        sequential.Recurrent(2, 'tanh'),
        seq_to_static,
        simple.AffineNonlinear(2, 3),
        seq_to_static.inverse,
    ]
    loss = simple.SupervisedLoss('squared', target=target)

    s = base.SupervisedStack(layers=layers, loss=loss)

    s.forward(inpt)

    s.function(['inpt', 'target'], 'loss')
    X = np.zeros((10, 5, 2))
    Z = np.zeros((10, 5, 3))
    Y = s.predict(X)
    assert Y.shape == (10, 5, 3), 'shape of output not right'
