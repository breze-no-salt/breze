# -*- coding: utf-8 -*-


import theano.tensor as T
from breze.arch import stack


def test_simple_stack():
    inpt = T.matrix('inpt')
    target = T.matrix('target')

    s = stack.Stack()

    layer = stack.AffineNonlinear(10, 2)
    layer.name = 'layer1'
    s.layers.append(layer)

    loss = stack.SupervisedLoss('squared', target)
    loss.name = 'loss'
    s.layers.append(loss)

    s.finalize(inpt)
    s.function(['inpt', ('loss', 'target')], ('loss', 'total'))

