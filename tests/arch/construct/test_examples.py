# -*- coding: utf-8 -*-


import theano.tensor as T

from breze.arch.construct.simple import AffineNonlinear


def test_linear():
    inpt = T.matrix('inpt')

    l = AffineNonlinear(inpt, 10, 2, 'tanh')

    assert l.inpt is inpt
    assert isinstance(l.output, T.TensorVariable)

    assert hasattr(l, 'weights')
    assert hasattr(l, 'bias')

    l = AffineNonlinear(inpt, 10, 2, 'tanh', use_bias=False)

    assert l.inpt is inpt
    assert isinstance(l.output, T.TensorVariable)

    assert hasattr(l, 'weights')
    assert not hasattr(l, 'bias')
