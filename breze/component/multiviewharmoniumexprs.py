# -*- coding: utf-8 -*-

import collections
import types

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from math import pi

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm


class MultiViewHarmoniumExprs(object):

    # Model hyperparameters:
    #
    # vis[view].f(x_vis[node, sample]) -> fv_vis[node, sample, statistic]
    # phid[view].f(x_phid[node, sample]) -> fv_phid[node, sample, statistic]
    # shid.f(x_shid[node, sample]) -> fv_shid[node, sample, statistic]
    #
    # vis[view].lp(fac[node, sample, statistic]) -> lpv_vis[node, sample]
    # phid[view].lp(fac[node, sample, statistic]) -> lpv_phid[node, sample]
    # shid.lp(fac[node, sample, statistic]) -> lpv_shid[node, sample]
    #
    # phid[view].dlp(fac[node, sample, statistic]) -> dlpv_phid[node, sample, statistic]
    # shid.dlp(fac[node, sample, statistic]) -> dlpv_shid[node, sample, statistic]
    #
    # vis[view].sampler(fac[node, sample, statistic]) -> sample_vis[node, sample]
    # phid[view].sampler(fac[node, sample, statistic]) -> sample_phid[node, sample]
    # shid.sampler(fac[node, sample, statistic]) -> sample_shid[node, sample]

    # Model parameters:
    # bias_vis[view][node, statistic]
    # bias_phid[view][node, statistic]
    # bias_shid[node, statistic]
    # weights_priv[view][to_node, to_statistic, from_node, from_statistic]
    # weights_shrd[view][to_node, to_statistic, from_node, from_statistic]

    # Inputs:
    # x_vis[view][node, sample]
    # x_phid[view][node, sample]
    # x_shid[node, sample]

    def __init__(self, 
                 vis, phid, shid, 
                 n_vis_nodes, n_phid_nodes, n_shid_nodes,
                 n_gs_learn,
                 bias_vis, bias_phid, bias_shid,
                 weights_priv, weights_shrd):
        # Hyperparameters
        self.vis = vis
        self.phid = phid
        self.shid = shid
        self.n_vis_nodes = n_vis_nodes
        self.n_phid_nodes = n_phid_nodes
        self.n_shid_nodes = n_shid_nodes
        self.n_gs_learn = n_gs_learn

        # Parameters
        self.bias_vis = bias_vis
        self.bias_phid = bias_phid
        self.bias_shid = bias_shid
        self.weights_priv = weights_priv
        self.weights_shrd = weights_shrd

    def check_dimensions(self):
        assert len(self.vis) == self.n_views
        assert len(self.phid) == self.n_views
        assert len(self.bias_vis) == self.n_views
        assert len(self.bias_phid) == self.n_views
        assert len(self.weights_priv) == self.n_views
        assert len(self.weights_shrd) == self.n_views

    @property
    def n_views(self):
        return len(self.vis)

    def fac_shid(self, x_vis):
        # calculate factor of shared hidden units
        # fac_shid[node, sample, statistic]

        n_samples = x_vis[0].shape[1]
        facv_shid = T.zeros((self.n_shid_nodes, 
                             n_samples, 
                             self.shid.n_statistics))
        for statistic in range(self.shid.n_statistics):
            facv_shid[:, :, statistic] = T.tile(self.bias_shid[:, statistic],
                                                (n_samples, 1)).T
            if self.shid.fixed_bias[statistic]:
                facv_shid[:, :, statistic] = self.shid.fixed_bias_value[statistic]

            for from_view in range(self.n_views):
                fv_vis = self.vis[from_view].f(x_vis[from_view])
                for from_statistic in range(self.vis[from_view].n_statistics):
                    facv_shid[:, :, statistic] += \
                        T.dot(self.weights_shrd[from_view][:, statistic, :, from_statistic].T,
                                                fv_vis[:, :, from_statistic])
        return facv_shid
            
    def p_shid(self, x_shid, x_vis):
        """Probability p_shid[node, sample] of shared hidden units having values 
            x_shid[node, sample] given that visible units have values 
            x_vis[view][node, sample]"""
        # p_shid[node, sample]
        fv_shid = self.shid.f(x_shid)
        facv_shid = self.fac_shid(x_vis)
        lpv_shid = self.shid.lp(facv_shid)
        return (facv_shid * fv_shid).sum(axis=2) - lpv_shid

    def sample_shid(self, x_vis):
        """Sample shared hidden units x_shid[node, sample] given that visible units 
            have values x_vis[view][node, sample]"""
        facv_shid = self.fac_shid(x_vis)
        return self.shid.sampler(facv_shid)

    def fac_phid(self, x_vis):
        # calculate probability of private hidden units
        # fac_phid[view][node, sample, statistic]
        n_samples = x_vis[0].shape[1]
        facv_phid = [T.zeros((self.n_phid_nodes[view],
                              n_samples,
                              self.phid[view].n_statistics)) 
                     for view in range(self.n_views)]
        for view in range(self.n_views):      
            fv_vis = self.vis[view].f(x_vis[view])
            for statistic in range(self.phid[view].n_statistics):
                facv_phid[view][:, :, statistic] = T.tile(self.bias_phid[view][:, statistic],
                                                          (n_samples, 1)).T
                if self.phid[view].fixed_bias[statistic]:
                    facv_phid[view][:, :, statistic] = self.phid[view].fixed_bias_value[statistic]

                for from_statistic in range(self.vis[view].n_statistics):
                    facv_phid[view][:, :, statistic] += \
                        T.dot(self.weights_priv[view][:, statistic, :, from_statistic].T,
                              fv_vis[:, :, from_statistic])
        return facv_phid

    def p_phid(self, x_phid, x_vis):
        """Probability p_phid[view][node, sample] of private hidden units having 
            values x_phid[view][node, sample] given that visible units have values 
            x_vis[view][node, sample]"""
        facv_phid = self.fac_phid(x_vis)
        pv_phid = []
        for view in range(self.n_views):
            fv_phid = self.phid[view].f(x_phid[view])
            lpv_phid = self.phid[view].lp(facv_phid[view])
            pv_phid.append((facv_phid[view] * fv_phid).sum(axis=2) - lpv_phid)
        return pv_phid

    def sample_phid(self, x_vis):
        """Sample private hidden units x_phid[view][node, sample] given that 
            visible units have values x_vis[view][node, sample]"""
        facv_phid = self.fac_phid(x_vis)
        samplev_phid = []
        for view in range(self.n_views):
            samplev_phid.append(self.phid[view].sampler(facv_phid[view]))
        return samplev_phid

    def fac_vis(self, x_phid, x_shid):
        # calculate probability of visible units
        # fac_vis[view][node, sample, statistic]
        n_samples = x_phid[0].shape[1]
        assert n_samples == x_shid.shape[1]
        facv_vis = [T.zeros((self.n_vis_nodes[view],
                             n_samples,
                             self.vis[view].n_statistics)) 
                    for view in range(self.n_views)]
        fv_shid = self.shid.f(x_shid)
        for view in range(self.n_views):      
            fv_phid = self.phid[view].f(x_phid[view])
            for statistic in range(self.vis[view].n_statistics):
                facv_vis[view][:, :, statistic] = T.tile(self.bias_vis[view][:, statistic],
                                                         (n_samples, 1)).T
                if self.vis[view].fixed_bias[statistic]:
                    facv_vis[view][:, :, statistic] = self.vis[view].fixed_bias_value[statistic]

                for from_statistic in range(self.phid[view].n_statistics):
                    facv_vis[view][:, :, statistic] += \
                        T.dot(self.weights_priv[view][:, statistic, :, from_statistic],
                              fv_phid[:, :, from_statistic])
                for from_statistic in range(self.shid.n_statistics):
                    facv_vis[view][:, :, statistic] += \
                        T.dot(self.weights_shrd[view][:, statistic, :, from_statistic],
                              fv_shid[:, :, from_statistic])
        return facv_vis

    def p_vis(self, x_vis, x_phid, x_shid):
        """Probability p_vis[view][node, sample] of visible units having values 
            x_vis[view][node, sample] given that private hidden units have values 
            x_phid[view][node, sample] and shared hidden units have values 
            x_shid[node, sample]"""
        facv_vis = self.fac_vis(x_phid, x_shid)
        pv_vis = []
        for view in range(self.n_views):
            fv_vis = self.vis[view].f(x_vis[view])
            lpv_vis = self.vis[view].lp(facv_vis[view])
            pv_vis.append((facv_vis[view] * fv_vis).sum(axis=2) - lpv_vis)
        return pv_vis

    def sample_vis(self, x_phid, x_shid):
        """Sample visible units x_vis[view][node, sample] given that private 
            hidden units have values x_phid[view][node, sample] and shared hidden 
            units have values x_shid[node, sample]"""
        facv_vis = self.fac_vis(x_phid, x_shid)
        samplev_vis = []
        for view in range(self.n_views):
            samplev_vis.append(self.vis[view].sampler(facv_vis[view]))
        return samplev_vis

    def gibbs_sample_vis(self, x_vis_start, x_phid_start, x_shid_start,
                         vis_clamp, phid_clamp, shid_clamp,
                         n_iterations):
        n_samples = x_vis_start.shape[1]
        x_vis = x_vis_start
        for i in range(n_iterations):
            # sample private hiddens given visibles
            x_phid = self.sample_phid(x_vis)
            if phid_clamp is not None:
                for view in range(self.n_views):
                    if phid_clamp[view]:
                        x_phid[view] = x_phid_start[view]

            # sample shared hiddens given visibles
            if not shid_clamp:
                x_shid = self.sample_shid(x_vis)
            else:
                x_shid = x_shid_start

            # sample visibles given hiddens
            x_vis = self.sample_vis(x_phid, x_shid)
            if vis_clamp is not None:
                for view in range(self.n_views):
                    if vis_clamp[view]:
                        x_vis = x_vis_start[view]

        return x_vis

    def _update_part(self, x_vis, distr): 
        to_multiview = isinstance(distr, collections.Iterable)
        if not to_multiview:
            facv = distr.fac(x_vis)
            dlpv = distr.dlp(facv)
            n_hid_statistics_view = distr.n_statistics

        # fv_vis[from_node, sample, from_statistic]
        # dlpv[to_node, sample, to_statistic]
        dbias_vis = []
        dbias_hid = []
        dweights = []
        for view in range(self.n_views):
            if to_multiview:
                facv = distr[view].fac(x_vis)
                dlpv = distr[view].dlp(facv)
                n_hid_statistics_view = distr[view].n_statistics

            fv_vis = self.vis[view].f(x_vis[view])

            dbias_vis.append(fv_vis.sum(axis=1))
            dbias_hid.append(dlpv.sum(axis=1))
            dweights.append(T.zeros((dlpv.shape[0], n_hid_statistics_view, 
                                     fv_vis.shape[1], self.vis[view].n_statistics)))

            for to_statistic in range(n_hid_statistics_view):
                for from_statistic in range(self.vis[view].n_statistics):
                    dweights[view][:, to_statistic, : from_statistic] = \
                        T.dot(dlpv[:, :, to_statistic], fv_vis[:, :, from_statistic].T)

        return (dbias_vis, dbias_hid, dweights)


    def cd_learning_update(self, x_vis, n_gibbs_iterations):

        # data distribution        
        data_dbias_vis, data_dbias_phid, data_dweights_priv = \
            self._update_part(x_vis, self.phid)
        _, data_dbias_shid, data_dweights_shrd = \
            self._update_part(x_vis, self.shid)

        # model distribution
        model_x_vis = self.gibbs_sample_vis(x_vis, None, None, None, None, None,
                                            n_gibbs_iterations)
        (model_dbias_vis, model_dbias_phid, model_dweights_priv) = \
            self._update_part(model_x_vis, self.phid)
        (_, model_dbias_shid, model_dweights_shrd) = \
            self._update_part(x_vis, self.shid)

        # compute CD parameter updates
        dbias_vis = [data_dbias_vis[view] - model_dbias_vis[view] 
                     for view in range(self.n_views)]
        dbias_phid = [data_dbias_phid[view] - model_dbias_phid[view] 
                      for view in range(self.n_views)]
        dbias_shid = data_dbias_shid - model_dbias_shid
        dweights_priv = [data_dweights_priv[view] - model_dweights_priv[view]
                         for view in range(self.n_views)]
        dweights_shrd = [data_dweights_shrd[view] - model_dweights_shrd[view]
                         for view in range(self.n_views)]

        return (dbias_vis, dbias_phid, dbias_shid,
                dweights_priv, dweights_shrd)

