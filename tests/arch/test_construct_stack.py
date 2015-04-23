# -*- coding: utf-8 -*-


import theano.tensor as T
from breze.arch.construct import stack


def test_simple_stack():
    s = stack.SupervisedStack()

    layer = stack.AffineNonlinear(10, 2)
    #layer.name = 'layer1'
    s.layers.append(layer)

    loss = stack.SupervisedLoss('squared')
    #loss.name = 'loss'
    s.loss = loss
    #s.layers.append(loss)

    s.finalize()
    s.function(['inpt', 'target'], 'loss')

