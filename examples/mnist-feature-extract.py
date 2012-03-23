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
from climin.stops import rising

from breze.model.feature import (
        SparseFiltering, ContractiveAutoEncoder, SparseAutoEncoder, Rica)
from breze.model.linear import Linear
from breze.model.neural import TwoLayerPerceptron

from utils import tile_raster_images, one_hot


# Hyperparameters.
batch_size = 1000
report_frequency = 50
max_iter_pretrain = 1000
max_iter_lr = 1000
max_iter_finetune = 1000
n_inpt = 784
n_feature = 256

method = 'cae'


# Make data.
f = gzip.open('mnist.pkl.gz', 'rb')
(X, Z), (VX, VZ), (TX, TZ) = cPickle.load(f)
f.close()

slices = draw_mini_slices(X.shape[0], batch_size)
args = (([X[s]], {}) for s in slices)


# Make feature extractor.
if method == 'sf':
    feature_transfer = 'softabs'
    fe = SparseFiltering(n_inpt, n_feature, feature_transfer)
    loss_key = 'loss'
    feature_key = 'feature'
    filter_key = 'inpt_to_feature'
elif method == 'rica':
    feature_transfer = 'softabs'
    fe = Rica(n_inpt, n_feature, feature_transfer, 'sigmoid', 
            'bernoulli_cross_entropy', c_ica=5)
    loss_key = 'loss'
    feature_key = 'feature'
    filter_key = 'in_to_hidden'
elif method == 'cae':
    feature_transfer = 'sigmoid'
    fe = ContractiveAutoEncoder(
            n_inpt, n_feature, feature_transfer, 'sigmoid', 'bernoulli_cross_entropy',
            c_jacobian=1.5)
    loss_key, feature_key = 'loss_reg', 'hidden'
    filter_key = 'in_to_hidden'
elif method == 'sae':
    feature_transfer = 'sigmoid'
    fe = SparseAutoEncoder(
            n_inpt, n_feature, 
            feature_transfer, 'sigmoid', 'bernoulli_cross_entropy',
            c_sparsity=5, sparsity_loss='bernoulli_cross_entropy',
            sparsity_target=0.05)
    loss_key, feature_key = 'loss_reg', 'hidden'
    filter_key = 'in_to_hidden'
else:
    assert False, 'unknown feature extractor'


# Compile functions.
f = fe.function(['inpt'], loss_key, explicit_pars=True)
d_loss_wrt_pars = T.grad(fe.exprs[loss_key], fe.parameters.flat)
fprime = fe.function(['inpt'], d_loss_wrt_pars, explicit_pars=True)
f_feature = fe.function(['inpt'], feature_key)


# Randomly initialize parameteres.
fe.parameters.data[:] = np.random.normal(0, 0.01, size=fe.parameters.data.shape)


print '--- PRETRAINING ---'
# Optimize!
opt = Lbfgs(fe.parameters.data, f, fprime, args=args)

for i, info in enumerate(opt):
    if i > max_iter_pretrain:
        break

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


print '--- TRAINING LOGISTIC REGRESSION LAYER ON TOP ---'

# Define logistic regression to see how that goes.
F, VF, TF = f_feature(X), f_feature(VX), f_feature(TX)
lr = Linear(n_feature, 10, 'softmax', 'cross_entropy')
loss = lr.exprs['loss']
d_loss_d_pars = T.grad(loss, lr.parameters.flat)

f = lr.function(['inpt', 'target'], 'loss', explicit_pars=True)
fprime = lr.function(['inpt', 'target'], d_loss_d_pars, explicit_pars=True)

Z2 = one_hot(Z, 10)
VZ2 = one_hot(VZ, 10)
TZ2 = one_hot(TZ, 10)

args = (([F, Z2], {}) for _ in itertools.count())
lr.parameters.data[:] = np.random.normal(0, 0.01, size=lr.parameters.data.shape)
f_predict = lr.function(['inpt'], T.argmax(lr.exprs['output_in'], axis=1))
opt = Lbfgs(lr.parameters.data, f, fprime, args=args)

f_val_loss = lambda: f(lr.parameters.data, VF, VZ2)
is_val_rising = rising(f_val_loss)

for i, info in enumerate(opt):
    loss = f(lr.parameters.data, F, Z2)
    val_loss = f(lr.parameters.data, VF, VZ2)
    test_loss = f(lr.parameters.data, TF, TZ2)
    print 'loss', loss, 'validate loss', val_loss, 'test loss', test_loss

    if is_val_rising(info):
        print 'loss rising on validation set'
        break

    if i > max_iter_lr:
        break

print 'lr empirical on train', (f_predict(F) == Z).mean()
print 'lr empirical on val', (f_predict(VF) == VZ).mean()
print 'lr empirical on test', (f_predict(TF) == TZ).mean()


print '--- FINE TUNING  ---'

# Define an mlp and fine tune it.
net = TwoLayerPerceptron(
    n_inpt, n_feature, 10, feature_transfer, 'softmax', 'cross_entropy')
net.parameters['in_to_hidden'][:] = fe.parameters[filter_key]
if 'hidden_bias' in fe.parameters:
    net.parameters['hidden_bias'] = fe.parameters['hidden_bias']
else:
    net.parameters['hidden_bias'][:] = np.random.normal(0, 0.001, size=n_features)
net.parameters['hidden_to_out'][:] = lr.parameters['in_to_out']
net.parameters['out_bias'][:] = lr.parameters['bias']

loss = net.exprs['loss']
d_loss_d_pars = T.grad(loss, net.parameters.flat)

f = net.function(['inpt', 'target'], 'loss', explicit_pars=True)
fprime = net.function(['inpt', 'target'], d_loss_d_pars, explicit_pars=True)
f_predict = net.function(['inpt'], T.argmax(net.exprs['output_in'], axis=1))
args = (([X, Z2], {}) for _ in itertools.count())

opt = Lbfgs(net.parameters.data, f, fprime, args=args)

f_val_loss = lambda: f(net.parameters.data, VX, VZ2)
is_val_rising = rising(f_val_loss)

for i, info in enumerate(opt):
    loss = f(net.parameters.data, X, Z2)
    val_loss = f(net.parameters.data, VX, VZ2)
    test_loss = f(net.parameters.data, TX, TZ2)
    print 'loss', loss, 'validate loss', val_loss, 'test loss', test_loss

    if is_val_rising(info):
        print 'loss rising on validation set'
        break

    if i > max_iter_finetune:
        break

print 'mlp empirical on train', (f_predict(X) == Z).mean()
print 'mlp empirical on val', (f_predict(VX) == VZ).mean()
print 'mlp empirical on test', (f_predict(TX) == TZ).mean()
