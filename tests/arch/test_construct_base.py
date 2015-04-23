# -*- coding: utf-8 -*-


import numpy as np

import theano.tensor as T
from breze.arch.construct import base


X = np.zeros((10, 2))
Z = np.zeros((10, 3))


def test_simple_stack():
    inpt = T.matrix('inpt')
    target = T.matrix('target')

    layers = [
        base.AffineNonlinear(2, 2),
        base.AffineNonlinear(2, 2),
        base.AffineNonlinear(2, 3),
    ]
    loss = base.SupervisedLoss('squared', target=target)

    s = base.SupervisedStack(layers=layers, loss=loss)

    s.forward(inpt)

    s.function(['inpt', 'target'], 'loss')
    Y = s.predict(X)
    assert Y.shape == (10, 3), 'shape of output not right'
