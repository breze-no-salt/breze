# -*- coding: utf-8 -*-

import collections
import types

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from math import pi

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm
from ...component.multiviewharmoniumexprs import MultiViewHarmoniumExprs


class MultiViewHarmonium(Model):

    def __init__(self, 
                 vis_dist, phid_dist, shid_dist, 
                 n_vis_nodes, n_phid_nodes, n_shid_nodes,
                 n_samples, n_gs_learn):
        self.exprs = MultiViewHarmoniumExprs(vis_dist, phid_dist, shid_dist,
                                             n_vis_nodes, n_phid_nodes, n_shid_nodes,
                                             n_samples, n_gs_learn,
                                             None, None, None,
                                             None, None)

        super(MultiViewHarmonium, self).__init__()

    def init_pars(self):       
        parspec = self.get_parameter_spec(self.exprs)
        self.parameters = ParameterSet(**parspec)

        self.exprs.bias_vis = [self.parameters['bias_vis_%d' % view] 
                               for view in range(self.exprs.n_views)]
        self.exprs.bias_phid = [self.parameters['bias_phid_%d' % view]
                                for view in range(self.exprs.n_views)]
        self.exprs.bias_shid = self.parameters['bias_shid']
        self.exprs.weights_priv = [self.parameters['weights_priv_%d' % view]
                                   for view in range(self.exprs.n_views)]
        self.exprs.weights_shrd = [self.parameters['weights_shrd_%d' % view]
                                   for view in range(self.exprs.n_views)]

    def init_exprs(self):
        self.x_vis = [T.matrix('x_vis_%d' % view) for view in range(self.exprs.n_views)]
        self.x_phid = [T.matrix('x_phid_%d' % view) for view in range(self.exprs.n_views)]
        self.x_shid = T.matrix('x_shid')

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

