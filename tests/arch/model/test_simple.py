# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from breze.arch.construct.simple import AffineNonlinear
from breze.arch.component.loss import squared
from breze.learn.base import SupervisedModel
from breze.utils.testhelpers import use_test_values


@use_test_values('raise')
def test_linear_regression():
    inpt = T.matrix('inpt')
    inpt.tag.test_value = np.zeros((3, 10))
    inpt.tag.test_value
    target = T.matrix('target')
    target.tag.test_value = np.zeros((3, 2))

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
