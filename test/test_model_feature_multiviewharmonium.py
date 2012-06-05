# -*- coding: utf-8 -*-

import cPickle
import itertools
import gzip
import sys

import Image as pil
import numpy as np
import theano
import theano.tensor as T

from breze.model.feature.multiviewharmonium import MultiViewHarmonium
from breze.util import WarnNaNMode
from breze.component.distribution import BernoulliDistribution
from breze.component.distribution import NormalDistribution

def test_mvh():   
    theano.config.compute_test_value = 'raise'

    # model parameters
    vis_dist = [BernoulliDistribution(), BernoulliDistribution()]
    phid_dist = [BernoulliDistribution(), BernoulliDistribution()]
    shid_dist = BernoulliDistribution()
    n_vis_nodes = [784, 384]
    n_phid_nodes = [512, 128]
    n_shid_nodes = 75
    n_gs_learn = 3

    # learning parameters
    batch_size = 20
    step_rate = 1E-1

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

    # test functions
    x_vis = [np.random.random((nodes, batch_size)) for nodes in n_vis_nodes]

    bias_vis_step, bias_phid_step, bias_shid_step, \
        weights_priv_step, weights_shrd_step = f_cd_learn(x_vis)
    for view in range(mvh.exprs.n_views):
        mvh.parameters['bias_vis_%d' % view] += step_rate * bias_vis_step[view]
        mvh.parameters['bias_phid_%d' % view] += step_rate * bias_phid_step[view]
        mvh.parameters['weights_priv_%d' % view] += step_rate * weights_priv_step[view]
        mvh.parameters['weights_shrd_%d' % view] += step_rate * weights_shrd_step[view]
    mvh.parameters['bias_shid'] += step_rate * bias_shid_step

    fac_phid = f_fac_phid(x_vis)
    fac_shid = f_fac_shid(x_vis)
    x_phid, x_shid = f_sample_hid(x_vis)
    x_vis = f_sample_vis(x_phid, x_shid)


def after_test():
    theano.config.compute_test_value = 'off'

test_mvh.teardown = after_test
