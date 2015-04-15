# -*- coding: utf-8 -*-


import theano.tensor as T
from breze.arch import stack


def test_simple_stack():
    s = stack.Stack()

    layer = stack.AffineNonlinear(10, 2)
    layer.name = 'layer1'
    s.layers.append(layer)

    inpt = T.matrix()
    target = T.matrix()

    loss = stack.SupervisedLoss('squared', target)
    loss.name = 'loss'

    s.layers.append(loss)

    s.finalize(inpt)

    s.function([inpt, target], loss.exprs['total'])

