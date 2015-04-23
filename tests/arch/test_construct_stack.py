# -*- coding: utf-8 -*-


import numpy as np

import theano.tensor as T
from breze.arch.construct import stack


X = np.zeros((10, 2))
Z = np.zeros((10, 3))


def test_simple_stack():
    layers = [
        stack.AffineNonlinear(2, 2),
        stack.AffineNonlinear(2, 2),
        stack.AffineNonlinear(2, 3),
    ]
    loss = stack.SupervisedLoss('squared')

    s = stack.SupervisedStack(layers=layers, loss=loss)

    inpt = T.matrix('inpt')
    s.forward(inpt)

    s.function(['inpt', 'target'], 'loss')
    Y = s.predict(X)
    assert Y.shape == (10, 3), 'shape of output not right'
