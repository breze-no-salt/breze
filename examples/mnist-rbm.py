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

from breze.model.feature import RestrictedBoltzmannMachine as RBM
from breze.util import WarnNaNMode

from utils import tile_raster_images, one_hot


# Hyperparameters.
batch_size = 20
n_feature = 512
step_rate = 1E-1
momentum = 0.9
n_gibbs_steps = 1

# Make data.
f = gzip.open('mnist.pkl.gz', 'rb')
(X, Z), (VX, VZ), (TX, TZ) = cPickle.load(f)
f.close()

slices = draw_mini_slices(X.shape[0], batch_size)
args = (X[s] for s in slices)

# Build RBM
rbm = RBM(784, n_feature) 
f_feature_sample = rbm.function(
    ['inpt', 'n_gibbs_steps'], 
    ['feature_sample', 'gibbs_sample_visible'],
    )
f_p_feature_given_inpt = rbm.function(['inpt'], 'p_feature_given_inpt')

rbm.parameters.data[:] = np.random.normal(0, 1E-2, rbm.parameters.data.shape)


def learn_step(x):
    n = x.shape[0]
    feature_sample, recons = f_feature_sample(x, n_gibbs_steps)
    recons_features = f_p_feature_given_inpt(recons)
    in_to_feature_step = (np.dot(x.T, feature_sample) - np.dot(recons.T, recons_features))
    in_bias_step = (x - recons).mean(axis=0)
    feature_bias_step = (feature_sample - recons_features).mean(axis=0)

    return in_to_feature_step / n, in_bias_step, feature_bias_step

in_to_feature_update_m1 = 0
in_bias_update_m1 = 0
feature_bias_update_m1 = 0

for i in range(10000):
    print '.'
    x = args.next()

    in_to_feature_step, in_bias_step, feature_bias_step = learn_step(x)

    in_to_feature_update = momentum * in_to_feature_update_m1 + step_rate * in_to_feature_step
    in_bias_update = momentum * in_bias_update_m1 + step_rate * in_bias_step
    feature_bias_update = momentum * feature_bias_update_m1 + step_rate * feature_bias_step
    
    rbm.parameters['in_to_feature'] += in_to_feature_update
    rbm.parameters['in_bias'] += in_bias_step
    rbm.parameters['feature_bias'] += feature_bias_step

    in_to_feature_update_m1 = in_to_feature_update
    in_bias_update_m1 = in_bias_update
    feature_bias_update_m1 = feature_bias_update

    if i % 100 == 0:
        W = rbm.parameters['in_to_feature'][:]
        A = tile_raster_images(W.T, (28, 28), (8, 8)).astype('float64')
        pilimage = pil.fromarray(A).convert('RGB')
        pilimage.save('rbm-mnist-filters-%03i.png' % i)

        A = tile_raster_images(x, (28, 28), (8, 8)).astype('float64')
        pilimage = pil.fromarray(A).convert('RGB')
        pilimage.save('rbm-mnist-inpts-%03i.png' % i)
