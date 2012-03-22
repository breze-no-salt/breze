# -*- coding: utf-8 -*-

import cPickle
import itertools
import gzip

import Image as pil
import numpy as np
import theano
import theano.tensor as T

from climin import Lbfgs
from climin.util import draw_mini_slices

from breze.model.feature import (
        SparseFiltering, ContractiveAutoEncoder, SparseAutoEncoder, Rica)

from utils import tile_raster_images


# Hyperparameters.
batch_size = 1000
report_frequency = 50
max_iter = 1000
n_inpt = 784
n_feature = 64

method = 'cae'


# Make data.
f = gzip.open('mnist.pkl.gz', 'rb')
(X, Z), (VX, VZ), (TX, TZ) = cPickle.load(f)
f.close()

slices = draw_mini_slices(X.shape[0], batch_size)
args = (([X[s]], {}) for s in slices)


# Make feature extractor.
if method == 'sf':
    fe = SparseFiltering(n_inpt, n_feature, 'softabs')
    loss_key = 'loss'
    feature_key = 'feature'
    filter_key = 'inpt_to_feature'
elif method == 'rica':
    fe = Rica(n_inpt, n_feature, 'softabs', 'sigmoid', 
            'bernoulli_cross_entropy', c_ica=5)
    loss_key = 'loss'
    feature_key = 'feature'
    filter_key = 'inpt_to_hidden'
elif method == 'cae':
    fe = ContractiveAutoEncoder(
            n_inpt, n_feature, 'sigmoid', 'sigmoid', 'bernoulli_cross_entropy',
            c_jacobian=1000 / 64. * 0.1)
    loss_key, feature_key = 'loss_reg', 'hidden'
    filter_key = 'inpt_to_hidden'
elif method == 'sae':
    fe = SparseAutoEncoder(
            n_inpt, n_feature, 'sigmoid', 'sigmoid', 'bernoulli_cross_entropy',
            c_sparsity=(5 * n_feature), sparsity_loss='bernoulli_cross_entropy',
            sparsity_target=0.05)
    loss_key, feature_key = 'loss_reg', 'hidden'
    filter_key = 'inpt_to_hidden'
else:
    assert False, 'unknown feature extractor'


# Compile functions.
f = fe.function(['inpt'], loss_key, explicit_pars=True)
d_loss_wrt_pars = T.grad(fe.exprs[loss_key], fe.parameters.flat)
fprime = fe.function(['inpt'], d_loss_wrt_pars, explicit_pars=True)


# Randomly initialize parameteres.
fe.parameters.data[:] = np.random.normal(0, 0.01, size=fe.parameters.data.shape)


# Optimize!
opt = Lbfgs(fe.parameters.data, f, fprime, args=args)
for i, info in enumerate(opt):
    if i % report_frequency != 0:
        continue

    loss = f(fe.parameters.data, X)
    val_loss = f(fe.parameters.data, VX)
    test_loss = f(fe.parameters.data, TX)
    print 'loss', loss, 'validate loss', val_loss, 'test loss', test_loss

    # Visualize filters.
    W = fe.parameters[filter_key]
    A = tile_raster_images(W.T, (28, 28), (8, 8)).astype('float64')
    pilimage = pil.fromarray(A).convert('RGB')
    pilimage.save('%s-mnist-filters-%i.png' % (method, i))

    if i > max_iter:
        break
