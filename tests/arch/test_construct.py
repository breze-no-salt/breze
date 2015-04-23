# -*- coding: utf-8 -*-


import numpy as np

import theano.tensor as T
from breze.arch.construct import base
from breze.arch.construct.layer  import simple


X = np.zeros((10, 2))
Z = np.zeros((10, 3))


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
