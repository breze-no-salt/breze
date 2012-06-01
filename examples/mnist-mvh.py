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

from breze.model.feature.multiviewharmonium import MultiViewHarmonium
from breze.util import WarnNaNMode
from breze.component.distribution import BernoulliDistribution
from breze.component.distribution import NormalDistribution

from utils import tile_raster_images, one_hot


# model parameters
vis_dist = [BernoulliDistribution()]
phid_dist = [BernoulliDistribution()]
shid_dist = BernoulliDistribution()
n_vis_nodes = [784]
n_phid_nodes = [512]
n_shid_nodes = 0
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

# Build MVH
mvh = MultiViewHarmonium(vis_dist, phid_dist, shid_dist,
                         n_vis_nodes, n_phid_nodes, n_shid_nodes,
                         batch_size, n_gs_learn)
#f_cd_learn = theano.function([mvh.x_vis[0]], 
#                             [
f_cd_learn = mvh.function(mvh.x_vis,
                          mvh.exprs.cd_learning_update(mvh.x_vis))
mvh.parameters.data[:] = np.random.normal(0, 1E-2, mvh.parameters.data.shape)

for i in range(10000):
    sys.stdout.write('.')
    x_vis = args.next()

    bias_vis_step, bias_phid_step, bias_shid_step, \
       weights_priv_step, weights_shrd_step = f_cd_learn(x_vis)

    for view in mvh.exprs.n_views:
        mvh.exprs.bias_vis[view] += step_rate * bias_vis_step[view]
        mvh.exprs.bias_phid[view] += step_rate * bias_phid_step[view]
        mvh.exprs.weights_priv[view] += step_rate * weights_priv_step[view]
        mvh.exprs.weights_shrd[view] += step_rate * weights_shrd_step[view]
    mvh.exprs.bias_shid += step_rate * bias_shid_step

    if i % 100 == 0:
        W = mvh.exprs.weights_priv[0][:, 0, :, 0]
        A = tile_raster_images(W.T, (28, 28), (8, 8)).astype('float64')
        pilimage = pil.fromarray(A).convert('RGB')
        pilimage.save('mvh-mnist-filters-%05i.png' % i)

