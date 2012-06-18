# -*- coding: utf-8 -*-

import collections
import types

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from math import pi

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm, multiviewharmonium


class MultiViewHarmonium(Model):
    """Multi-View Harmonium model
    
    See breze.component.multiviewharmonium.MultiViewHarmonium for
    documentation about the implementation of the model.
    """

    """Multi-View Harmonium model

    :param vis_dist: list of distributions for visible nodes of each view
    :param phid_dist: list of distributions for private latent nodes of each view
    :param shid_dist: distribution of shared latent nodes
    :param n_vis_nodes: list of number of visible nodes for each view
    :param n_phid_nodes: list of number of private latent nodes for each view
    :param n_shid_nodes: number of shared latent nodes
    :param n_samples: number of samples per mini-batch
    :param n_gs_learn: number of Gibbs iterations for contrastive-divergence 
                       learning
    """
    def __init__(self, 
                 vis_dist, phid_dist, shid_dist, 
                 n_vis_nodes, n_phid_nodes, n_shid_nodes,
                 n_samples, n_gs_learn):
        self.exprs = multiviewharmonium.MultiViewHarmonium(
            vis_dist, phid_dist, shid_dist,
            n_vis_nodes, n_phid_nodes, n_shid_nodes,
            n_samples, n_gs_learn,
            None, None, None,
            None, None)

        super(MultiViewHarmonium, self).__init__()

    def init_pars(self):       
        parspec = self.get_parameter_spec(self.exprs)
        self.parameters = ParameterSet(**parspec)

        self.exprs.bias_vis = [getattr(self.parameters, 'bias_vis_%d' % view)
                               for view in range(self.exprs.n_views)]
        self.exprs.bias_phid = [getattr(self.parameters, 'bias_phid_%d' % view)
                                for view in range(self.exprs.n_views)]
        self.exprs.bias_shid = getattr(self.parameters, 'bias_shid')
        self.exprs.weights_priv = [getattr(self.parameters, 'weights_priv_%d' % view)
                                   for view in range(self.exprs.n_views)]
        self.exprs.weights_shrd = [getattr(self.parameters, 'weights_shrd_%d' % view)
                                   for view in range(self.exprs.n_views)]

    def init_exprs(self):
        self.x_vis = [T.matrix('x_vis_%d' % view) for view in range(self.exprs.n_views)]
        self.x_phid = [T.matrix('x_phid_%d' % view) for view in range(self.exprs.n_views)]
        self.x_shid = T.matrix('x_shid')

        self.updates = collections.defaultdict(lambda: {})

    @staticmethod
    def get_parameter_spec(exprs):
        ps = {'bias_shid': (exprs.n_shid_nodes, exprs.shid.n_statistics)}
        for view in range(exprs.n_views):
            ps['bias_vis_%d' % view] = (exprs.n_vis_nodes[view], exprs.vis[view].n_statistics)
            ps['bias_phid_%d' % view] = (exprs.n_phid_nodes[view], exprs.phid[view].n_statistics)
            ps['weights_priv_%d' % view] = (exprs.n_phid_nodes[view], exprs.phid[view].n_statistics,
                                            exprs.n_vis_nodes[view], exprs.vis[view].n_statistics)
            ps['weights_shrd_%d' % view] = (exprs.n_shid_nodes, exprs.shid.n_statistics,
                                            exprs.n_vis_nodes[view], exprs.vis[view].n_statistics)
        return ps

