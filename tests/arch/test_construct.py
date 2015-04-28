# -*- coding: utf-8 -*-


import numpy as np

import theano
import theano.tensor as T
from breze.arch.construct import base
from breze.arch.construct.layer import simple, sequential


X = np.zeros((10, 2)).astype(theano.config.floatX)
Z = np.zeros((10, 3)).astype(theano.config.floatX)


theano.config.compute_test_value = 'raise'


def test_simple_stack():
    inpt = T.matrix('inpt')
    inpt.tag.test_value = X
    target = T.matrix('target')
    target.tag.test_value = Z

    layers = [
        simple.AffineNonlinear(2, 2),
        simple.AffineNonlinear(2, 2),
        simple.AffineNonlinear(2, 3),
    ]
    loss = simple.SupervisedLoss('squared', target=target)

    s = base.SupervisedStack(layers=layers, loss=loss)

    s.forward(inpt)
    s._replace_param_dummies()

    s.function(['inpt', 'target'], 'loss')
    Y = s.predict(X)
    assert Y.shape == (10, 3), 'shape of output not right'


def test_sequential():
    inpt = T.tensor3('inpt')
    inpt.tag.test_value = np.zeros((2, 3, 4)).astype(theano.config.floatX)
    seq_to_static = sequential.SequentialToStatic()
    output = seq_to_static(inpt)
    recons = seq_to_static.inverse(*output)

    assert output[0].tag.test_value.shape == (6, 4)
    assert recons[0].tag.test_value.shape == (2, 3, 4)


def test_rnn():
    inpt = T.tensor3('inpt')
    inpt.tag.test_value = np.empty((10, 5, 2)).astype(theano.config.floatX)

    target = T.tensor3('target')
    target.tag.test_value = np.empty((10, 5, 3)).astype(theano.config.floatX)

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
    s._replace_param_dummies()

    s.function(['inpt', 'target'], 'loss')
    X = np.zeros((10, 5, 2)).astype(theano.config.floatX)
    Y = s.predict(X)
    assert Y.shape == (10, 5, 3), 'shape of output not right'


def test_supervised_loss():
    prediction = T.matrix('prediction')
    y = prediction.tag.test_value = np.zeros((2, 3)).astype(theano.config.floatX)


    target = T.matrix('target')
    z = target.tag.test_value = prediction.tag.test_value.copy()
    z[0, 0] -= 1

    loss_layer = simple.SupervisedLoss('squared', target)
    loss_expr, = loss_layer(prediction)

    l = loss_expr.eval({prediction: y, target: z})
    assert l == 1. / y.shape[0], 'loss calculation wrong'

    imp_weight = T.matrix('imp_weight')
    w = np.zeros_like(y).astype(theano.config.floatX)
    imp_weight.tag.test_value = w
    w[0, 0,] = 1

    loss_layer = simple.SupervisedLoss('squared', target, imp_weight=imp_weight)
    loss_expr, = loss_layer(prediction)

    l = loss_expr.eval({prediction: y, target: z, imp_weight: w})
    assert l == 1. / y.shape[0], 'loss calculation with imp weights wrong'

    l = loss_expr.eval({prediction: y,
                        target: z,
                        imp_weight:
                            np.zeros_like(w).astype(theano.config.floatX)
})
    assert l == 0, 'loss calculation with imp weights wrong'

    l = loss_expr.eval({prediction: y,
                        target: z,
                        imp_weight:
                            np.ones_like(w).astype(theano.config.floatX)})
    assert l == 1. / y.shape[0], 'loss calculation with imp weights wrong'
