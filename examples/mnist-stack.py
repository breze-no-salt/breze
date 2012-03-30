# -*- coding: utf-8 -*-

import itertools

import numpy as np
import theano.tensor as T

from breze.model.feature import (
    SparseAutoEncoder, cascade_layers, get_affine_parameters,
    mlp_from_cascade, feature_func_from_model)
from breze.model.linear import Linear
from breze.model.neural import MultiLayerPerceptron
from breze.util import opt_from_model

from climin import Lbfgs
from climin.stops import rising, after_n_iterations, modulo_n_iterations
from climin.util import draw_mini_slices

from utils import one_hot


def train(n_inpt, layer_classes, layer_kwargs, opt_classes, opt_kwargs,
          batch_sizes, report_intervals, max_iters,
          X, Z, VX, VZ, TX, TZ):
    """Train a deep network by first building a stack of layers and optimizing
    each greedily, and afterwards fine tune it as an mlp."""

    layers = cascade_layers(n_inpt, layer_classes, layer_kwargs)

    #
    # Unsupervised Pretraining.
    #

    F, VF, TF = X, VX, TX
    for l, opt_class, opt_kwarg, b, r, mi in zip(
            layers[:-1], opt_classes, opt_kwargs, batch_sizes, report_intervals,
            max_iters):
        slices = draw_mini_slices(F.shape[0], b)
        us_args = (([F[s]], {}) for s in slices)

        opt = opt_from_model(l, ['inpt'], us_args, opt_class, opt_kwarg)

        is_rising = rising(lambda: opt.f(opt.wrt, F))
        max_iter_reached = after_n_iterations(mi)
        do_report = modulo_n_iterations(r)

        for i, info in enumerate(opt):
            if not do_report(info):
                continue

            print i, 'train loss', opt.f(opt.wrt, F),
            print 'val loss', opt.f(opt.wrt, VF)

            if is_rising(info):
                print 'val loss rising, break'
                break

            if max_iter_reached(info):
                print 'max iterations reached, break'
                break

        f = feature_func_from_model(l)
        F, VF, TF = f(F), f(VF), f(TF)
        us_args = (([F], {}) for _ in itertools.count())


    #
    # Supervised training of top layer.
    #

    slices = draw_mini_slices(F.shape[0], batch_sizes[-2])
    s_args = (([F[s], Z[s]], {}) for s in slices)
    opt = opt_from_model(
        layers[-1], ['inpt', 'target'], s_args, opt_classes[-2], opt_kwargs[-2])
    f_loss = layers[-1].function(['inpt', 'target'], 'loss')
    l = layers[-1]

    # For reporting.
    empirical = T.eq(T.argmax(l.exprs['output'], axis=1), 
                     T.argmax(l.exprs['target'], axis=1)).mean()
    f_empirical = l.function(['inpt', 'target'], empirical)

    # Stopping criterion. 
    is_rising = rising(lambda: opt.f(opt.wrt, F, Z))
    max_iter_reached = after_n_iterations(max_iters[-2])
    do_report = modulo_n_iterations(report_intervals[-2])

    for i, info in enumerate(opt):
        if not do_report(info):
            continue

        print i, 'train loss', opt.f(opt.wrt, F, Z), 
        print 'val loss', opt.f(opt.wrt, VF, VZ)
        print 'train empirical', f_empirical(F, Z),
        print 'val empirical', f_empirical(VF, VZ)

        if is_rising(info):
            print 'val loss rising, break'
            break

        if max_iter_reached(info):
            print 'max iterations reached, break'
            break


    #
    # Supervised tuning of complete stack.
    #

    slices = draw_mini_slices(X.shape[0], batch_sizes[-1])
    s_args = (([X[s], Z[s]], {}) for s in slices)
    mlp = mlp_from_cascade(layers, 'cross_entropy')

    f_loss = mlp.function(['inpt', 'target'], 'loss')
    opt = opt_from_model(mlp, ['inpt', 'target'], s_args,
                         opt_classes[-1], opt_kwargs[-1])

    # For reporting.
    empirical = T.eq(T.argmax(mlp.exprs['output'], axis=1), 
                     T.argmax(mlp.exprs['target'], axis=1)).mean()
    f_empirical = mlp.function(['inpt', 'target'], empirical)

    # Stopping criterion.
    is_rising = rising(lambda: opt.f(opt.wrt, X, Z))
    max_iter_reached = after_n_iterations(max_iters[-1])
    do_report = modulo_n_iterations(report_intervals[-1])

    for i, info in enumerate(opt):
        if not do_report(info):
            continue

        print i, 'train loss', opt.f(opt.wrt, X, Z), 
        print 'val loss', opt.f(opt.wrt, VX, VZ),
        print 'train empirical', f_empirical(X, Z),
        print 'val empirical', f_empirical(VX, VZ)
        print 'test empirical', f_empirical(TX, TZ)

        if is_rising(info):
            print 'val loss rising, break'
            break

        if max_iter_reached(info):
            print 'max iterations reached, break'
            break

if __name__ == '__main__':
    import cPickle
    import gzip

    # Make data.
    f = gzip.open('mnist.pkl.gz', 'rb')
    (X, Z), (VX, VZ), (TX, TZ) = cPickle.load(f)
    f.close()
    Z = one_hot(Z, 10)
    VZ = one_hot(VZ, 10)
    TZ = one_hot(TZ, 10)

    layer_classes = [SparseAutoEncoder] * 2 + [Linear]
    layer_kwargs = [
            {'n_hidden': 512,
             'loss': 'bernoulli_cross_entropy',
             'sparsity_target': 0.05,
             'c_sparsity': 5,
             'sparsity_loss': 'bernoulli_cross_entropy',
             'hidden_transfer': 'sigmoid',
             'out_transfer': 'sigmoid'},
            {'n_hidden': 512,
             'loss': 'bernoulli_cross_entropy',
             'sparsity_target': 0.05,
             'c_sparsity': 5,
             'sparsity_loss': 'bernoulli_cross_entropy',
             'hidden_transfer': 'sigmoid',
             'out_transfer': 'sigmoid'},
            {'n_output': 10,
             'loss': 'cross_entropy',
             'out_transfer': 'softmax'}
    ]

    opt_classes = [Lbfgs] * 4
    opt_kwargs = [{}] * 4

    batch_sizes = [1000, 1000, 50000, 50000]
    report_intervals = [50] * 4

    max_iters = [1, 1, 1, 1]

    train(784, layer_classes, layer_kwargs, opt_classes, opt_kwargs,
          batch_sizes, report_intervals, max_iters,
          X, Z, VX, VZ, TX, TZ)
