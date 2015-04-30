# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from breze.arch.construct.simple import AffineNonlinear
from breze.arch.component.loss import squared
from breze.arch.model.simple import SupervisedModel


def test_linear_regression():
    inpt = T.matrix('inpt')
    target = T.matrix('target')

    l = AffineNonlinear(inpt, 10, 2, 'tanh')

    loss = squared(target, l.output).sum(1).mean()

    m = SupervisedModel(inpt=inpt, target=target, output=l.output, loss=loss,
                        parameters=l.parameters)

    f_predict = m.function([m.inpt], m.output)
    f_loss = m.function([m.inpt, m.target], m.loss)

    X = np.zeros((20, 10))
    Z = np.zeros((20, 2))

    Y = f_predict(X)

    assert Y.shape == (20, 2), 'ouput has wrong shape'

    l = f_loss(X, Z)

    assert np.array(l).ndim == 0, 'loss is not a scalar'
