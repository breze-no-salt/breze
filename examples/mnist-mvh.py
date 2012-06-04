# -*- coding: utf-8 -*-

import cPickle
import itertools
import gzip
import sys

import Image as pil
import numpy as np
import theano
import theano.tensor as T

from climin import Lbfgs
from climin.util import draw_mini_slices

from breze.model.feature.rbm import RestrictedBoltzmannMachine
from breze.model.feature.multiviewharmonium import MultiViewHarmonium
from breze.util import WarnNaNMode
from breze.component.distribution import BernoulliDistribution
from breze.component.distribution import NormalDistribution

from utils import tile_raster_images, one_hot

theano.config.compute_test_value = 'off'

# model parameters
vis_dist = [BernoulliDistribution()]
phid_dist = [BernoulliDistribution()]
shid_dist = BernoulliDistribution()
n_vis_nodes = [784]
n_phid_nodes = [512]
n_shid_nodes = 1
n_gs_learn = 1

# learning parameters
batch_size = 20
step_rate = 1E-1

# Make data.
f = gzip.open('mnist.pkl.gz', 'rb')
(X, Z), (VX, VZ), (TX, TZ) = cPickle.load(f)
f.close()
slices = draw_mini_slices(X.shape[0], batch_size)
args = (X[s] for s in slices)

# Build RBM (for comparison)
rbm = RestrictedBoltzmannMachine(n_vis_nodes[0], n_phid_nodes[0]) 
f_feature_sample = rbm.function(
    ['inpt', 'n_gibbs_steps'], 
    ['feature_sample', 'gibbs_sample_visible'],
    )
f_p_feature_given_inpt = rbm.function(['inpt'], 'p_feature_given_inpt')
rbm.parameters.data[:] = np.random.normal(0, 1E-2, rbm.parameters.data.shape)

def rbm_learn_step(x):
    n = x.shape[0]
    feature_sample, recons = f_feature_sample(x, n_gs_learn)
    x_features = f_p_feature_given_inpt(x)
    recons_features = f_p_feature_given_inpt(recons)
    in_to_feature_step = (np.dot(x.T, x_features) - np.dot(recons.T, recons_features))
    #in_to_feature_step = (np.dot(x.T, feature_sample) - np.dot(recons.T, recons_features))
    in_bias_step = (x - recons).mean(axis=0)
    feature_bias_step = (feature_sample - recons_features).mean(axis=0)

    return in_to_feature_step / n, in_bias_step, feature_bias_step

def rbm_gibbs_sample(x):
    n = x.shape[0]
    feature_sample, recons = f_feature_sample(x, n_gs_learn)
    return recons

# Build MVH
mvh = MultiViewHarmonium(vis_dist, phid_dist, shid_dist,
                         n_vis_nodes, n_phid_nodes, n_shid_nodes,
                         batch_size, n_gs_learn)
mvh.parameters.data[:] = np.random.normal(0, 1E-2, mvh.parameters.data.shape)

# setup debug values
for view in range(mvh.exprs.n_views):
    mvh.x_vis[view].tag.test_value = \
        np.random.random((mvh.exprs.n_vis_nodes[view], mvh.exprs.n_samples))
    mvh.x_phid[view].tag.test_value = \
        np.random.random((mvh.exprs.n_phid_nodes[view], mvh.exprs.n_samples))
mvh.x_shid.tag.test_value = \
    np.random.random((mvh.exprs.n_shid_nodes, mvh.exprs.n_samples))

# construct functions
f_fac_phid = mvh.function(mvh.x_vis,
                          mvh.exprs.fac_phid(mvh.x_vis))
f_fac_shid = mvh.function(mvh.x_vis,
                          mvh.exprs.fac_shid(mvh.x_vis))
f_fac_vis = mvh.function([mvh.x_phid, mvh.x_shid],
                         mvh.exprs.fac_vis(mvh.x_phid, mvh.x_shid))

f_sample_hid = mvh.function(mvh.x_vis,
                            [mvh.exprs.sample_phid(mvh.x_vis),
                             mvh.exprs.sample_shid(mvh.x_vis)])
f_sample_vis = mvh.function([mvh.x_phid, mvh.x_shid],
                            mvh.exprs.sample_vis(mvh.x_phid, mvh.x_shid))
f_gibbs_sample_vis = mvh.function(mvh.x_vis,
                                  mvh.exprs.gibbs_sample_vis(mvh.x_vis,
                                                             None,
                                                             None,
                                                             None,
                                                             None,
                                                             None,
                                                             1))

f_cd_learn = mvh.function(mvh.x_vis,
                          mvh.exprs.cd_learning_update(mvh.x_vis))

x_vis = args.next().T
fac_phid = f_fac_phid([x_vis])
fac_shid = f_fac_shid([x_vis])
x_phid, x_shid = f_sample_hid(x_vis)
x_vis = f_sample_vis(x_phid, x_shid)

print "Precheck done"

x_ref = args.next().T

#for i in range(10000):
for i in range(300):
    sys.stdout.write('.')
    x_vis = args.next().T

    # train RBM
    in_to_feature_step, in_bias_step, feature_bias_step = rbm_learn_step(x_vis.T)  
    rbm.parameters['in_to_feature'] += step_rate * in_to_feature_step
    rbm.parameters['in_bias'] += step_rate * in_bias_step
    rbm.parameters['feature_bias'] += step_rate * feature_bias_step

    # train MVH
    bias_vis_step, bias_phid_step, bias_shid_step, \
       weights_priv_step, weights_shrd_step = f_cd_learn([x_vis])
    for view in range(mvh.exprs.n_views):
        mvh.parameters['bias_vis_%d' % view] += step_rate * bias_vis_step[view]
        mvh.parameters['bias_phid_%d' % view] += step_rate * bias_phid_step[view]
        mvh.parameters['weights_priv_%d' % view] += step_rate * weights_priv_step[view]
        mvh.parameters['weights_shrd_%d' % view] += step_rate * weights_shrd_step[view]
    mvh.parameters['bias_shid'] += step_rate * bias_shid_step

    if i % 100 == 0:
        # plot RBM
        W = rbm.parameters['in_to_feature'][:]
        A = tile_raster_images(W.T, (28, 28), (8, 8)).astype('float64')
        pilimage = pil.fromarray(A).convert('RGB')
        pilimage.save('rbm-mnist-filters-%05i.png' % i)

        A = tile_raster_images(x_ref.T, (28, 28), (8, 8)).astype('float64')
        pilimage = pil.fromarray(A).convert('RGB')
        pilimage.save('mnist-input-%05i.png' % i)

        x_sample = rbm_gibbs_sample(x_ref.T)
        A = tile_raster_images(x_sample, (28, 28), (8, 8)).astype('float64')
        pilimage = pil.fromarray(A).convert('RGB')
        pilimage.save('rbm-mnist-sample-%05i.png' % i)

        # plot MVH
        W = mvh.parameters['weights_priv_0'][:, 0, :, 0]
        A = tile_raster_images(W, (28, 28), (8, 8)).astype('float64')
        pilimage = pil.fromarray(A).convert('RGB')
        pilimage.save('mvh-mnist-filters-%05i.png' % i)

        x_sample = f_gibbs_sample_vis([x_ref])[0]
        A = tile_raster_images(x_sample.T, (28, 28), (8, 8)).astype('float64')
        pilimage = pil.fromarray(A).convert('RGB')
        pilimage.save('mvh-mnist-sample-%05i.png' % i)

print "Training done"
