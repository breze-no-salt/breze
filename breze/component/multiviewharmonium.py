# -*- coding: utf-8 -*-

import collections
import types

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from math import pi

from . import transfer, distance, norm


class MultiViewHarmonium(object):
    """Multi-View Harmonium expression builder

    Implementation of a multi-view harmonium as described in
    Y. Kang, S. Choi: Restricted Deep Belief Networks for Multi-View Learning,
    2011

    For an explanation of exponential family harmoniums see
    M. Welling, M. Rosen-Zvi, G. Hinton: Exponential Family Harmoniums with
    an Application to Information Retrieval

    This implementation is best unterstood by considering (6) in this paper.
    The sufficient statistics (i.e. f_ia(x_i)) and 
    the log-partition function A_i({θ_ia}) are 
    calculated by the methods f(...) and lp(...) of the supplied distributions.     
    The weight updates in (9) use the derivative of the log-partition funciton,
    which is calculated by the method dlp(...) of the supplied distributions.
    To allow for efficient sampling from the distributions this implementation
    does not calculate the PDFs but instead uses the sampler(...) method of the 
    supplied distributions. 
    """      
    
    # Shapes and signatures of instance variables
    # ===========================================
    #
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
    #
    # Model parameters:
    # bias_vis[view][node, statistic]
    # bias_phid[view][node, statistic]
    # bias_shid[node, statistic]
    # weights_priv[view][to_node, to_statistic, from_node, from_statistic]
    # weights_shrd[view][to_node, to_statistic, from_node, from_statistic]
    #
    # Inputs:
    # x_vis[view][node, sample]
    # x_phid[view][node, sample]
    # x_shid[node, sample]

    """Multi-View Harmonium expression builder

    :param vis_dist: list of distributions for visible nodes of each view
    :param phid_dist: list of distributions for private latent nodes of each view
    :param shid_dist: distribution of shared latent nodes
    :param n_vis_nodes: list of number of visible nodes for each view
    :param n_phid_nodes: list of number of private latent nodes for each view
    :param n_shid_nodes: number of shared latent nodes
    :param n_samples: number of samples per mini-batch
    :param n_gs_learn: number of Gibbs iterations for contrastive-divergence 
                       learning
    :param bias_vis: biases for visible nodes of each view. Form:
                     bias_vis[view][node, statistic]
    :param bias_phid: biases for private hidden nodes of each view. Form: 
                      bias_phid[view][node, statistic]
    :param bias_shid: biases for shared hidden nodes. Form:
                      bias_shid[node, statistic]
    :param weights_priv: weights between visible and private latent nodes
                         of each view. Form:
                         weights_priv[view][to_node, to_statistic, from_node, from_statistic]
    :param weights_shrd: weights between visible nodes of each view and the shared
                         latent nodes. Form:
                         weights_shrd[view][to_node, to_statistic, from_node, from_statistic]
    """
    def __init__(self, 
                 vis_dist, phid_dist, shid_dist, 
                 n_vis_nodes, n_phid_nodes, n_shid_nodes,
                 n_samples, n_gs_learn,
                 bias_vis, bias_phid, bias_shid,
                 weights_priv, weights_shrd):
        # Hyperparameters
        self.vis = vis_dist
        self.phid = phid_dist
        self.shid = shid_dist
        self.n_vis_nodes = n_vis_nodes
        self.n_phid_nodes = n_phid_nodes
        self.n_shid_nodes = n_shid_nodes
        self.n_samples = n_samples
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

    """Number of views"""
    @property
    def n_views(self):
        return len(self.vis)

    def fac_shid(self, x_vis):
        # calculate factor of shared hidden units
        # fac_shid[node, sample, statistic]

        facv_shid = T.zeros((self.n_shid_nodes, 
                             self.n_samples, 
                             self.shid.n_statistics),
                            dtype=theano.config.floatX)
        for statistic in range(self.shid.n_statistics):
            facv_shid = T.set_subtensor(facv_shid[:, :, statistic],
                                        self.bias_shid[:, statistic].dimshuffle(0, 'x'))                                               
            if self.shid.fixed_bias[statistic]:
                facv_shid = T.set_subtensor(facv_shid[:, :, statistic],
                                            self.shid.fixed_bias_value[statistic])

            for from_view in range(self.n_views):
                fv_vis = self.vis[from_view].f(x_vis[from_view])
                for from_statistic in range(self.vis[from_view].n_statistics):
                    facv_shid = T.set_subtensor(facv_shid[:, :, statistic],
                        T.dot(self.weights_shrd[from_view][:, statistic, :, from_statistic],
                                                fv_vis[:, :, from_statistic]))
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
        return self.shid.sampler(facv_shid, False)

    def fac_phid(self, x_vis):
        # calculate probability of private hidden units
        # fac_phid[view][node, sample, statistic]
        facv_phid = [T.zeros((self.n_phid_nodes[view],
                              self.n_samples,
                              self.phid[view].n_statistics),
                             dtype=theano.config.floatX) 
                     for view in range(self.n_views)]
        for view in range(self.n_views):      
            fv_vis = self.vis[view].f(x_vis[view])
            for statistic in range(self.phid[view].n_statistics):
                facv_phid[view] = T.set_subtensor(facv_phid[view][:, :, statistic],
                                                  self.bias_phid[view][:, statistic].dimshuffle(0, 'x'))
                if self.phid[view].fixed_bias[statistic]:
                    facv_phid[view] = T.set_subtensor(facv_phid[view][:, :, statistic],
                                                      self.phid[view].fixed_bias_value[statistic])

                for from_statistic in range(self.vis[view].n_statistics):
                    facv_phid[view] = T.inc_subtensor(facv_phid[view][:, :, statistic],
                        T.dot(self.weights_priv[view][:, statistic, :, from_statistic],
                              fv_vis[:, :, from_statistic]))
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
            samplev_phid.append(self.phid[view].sampler(facv_phid[view], False))
        return samplev_phid

    def fac_vis(self, x_phid, x_shid):
        # calculate probability of visible units
        # fac_vis[view][node, sample, statistic]

        facv_vis = [T.zeros((self.n_vis_nodes[view],
                             self.n_samples,
                             self.vis[view].n_statistics),
                            dtype=theano.config.floatX) 
                    for view in range(self.n_views)]
        fv_shid = self.shid.f(x_shid)
        for view in range(self.n_views):      
            fv_phid = self.phid[view].f(x_phid[view])
            for statistic in range(self.vis[view].n_statistics):
                facv_vis[view] = T.set_subtensor(facv_vis[view][:, :, statistic],
                                                 self.bias_vis[view][:, statistic].dimshuffle(0, 'x'))
                if self.vis[view].fixed_bias[statistic]:
                    facv_vis[view] = T.set_subtensor(facv_vis[view][:, :, statistic],
                                                     self.vis[view].fixed_bias_value[statistic])

                for from_statistic in range(self.phid[view].n_statistics):
                    facv_vis[view] = T.inc_subtensor(facv_vis[view][:, :, statistic], 
                        T.dot(self.weights_priv[view][:, statistic, :, from_statistic].T,
                              fv_phid[:, :, from_statistic]))
                for from_statistic in range(self.shid.n_statistics):
                    facv_vis[view] = T.inc_subtensor(facv_vis[view][:, :, statistic],
                        T.dot(self.weights_shrd[view][:, statistic, :, from_statistic].T,
                              fv_shid[:, :, from_statistic]))
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

    def sample_vis(self, x_phid, x_shid, final_gibbs_sample=False):
        """Sample visible units x_vis[view][node, sample] given that private 
            hidden units have values x_phid[view][node, sample] and shared hidden 
            units have values x_shid[node, sample]"""
        facv_vis = self.fac_vis(x_phid, x_shid)
        samplev_vis = []
        for view in range(self.n_views):
            samplev_vis.append(self.vis[view].sampler(facv_vis[view], 
                                                      final_gibbs_sample))
        return samplev_vis

    def gibbs_sample_vis(self, x_vis_start, x_phid_start, x_shid_start,
                         vis_clamp, phid_clamp, shid_clamp,
                         n_iterations):
        """Iteratively sample latent and visble units for n_iterations.

        The visible nodes are initialized with x_vis_start[view][node]. 

        The private nodes are then sampled for each view for which
        phid_clamp[view] is False; for the views for which phid_clamp[view] is
        True the value provided by x_phid_start[view][node] is used.
        The shared nodes are sampled if shid_clamp
        is False, otherwise the value provided by x_shid_start is used.

        Afterwards the visbles nodes are sampled for each view for
        which vis_clamp[view] ist False. The process is repeated n_iterations
        times.
        """

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
            x_vis = self.sample_vis(x_phid, x_shid, i == n_iterations - 1)
            if vis_clamp is not None:
                for view in range(self.n_views):
                    if vis_clamp[view]:
                        x_vis = x_vis_start[view]

        return x_vis

    def _update_part(self, x_vis): 
        facv_phid = self.fac_phid(x_vis)
        facv_shid = self.fac_shid(x_vis)

        dbias_vis = []
        dbias_phid = []
        dbias_shid = T.zeros(self.bias_shid.shape, dtype=theano.config.floatX)
        dweights_priv = []
        dweights_shrd = []
       
        for view in range(self.n_views):
            fv_vis = self.vis[view].f(x_vis[view])

            # update for private hidden weights
            dbias_vis_view, dbias_phid_view, dweights_priv_view = \
                self._update_part_view(fv_vis, facv_phid[view], 
                                       self.vis[view], self.phid[view])
            dbias_vis.append(dbias_vis_view)
            dbias_phid.append(dbias_phid_view)
            dweights_priv.append(dweights_priv_view)

            # update for shared hidden weights
            _, dbias_shid_view, dweights_shrd_view = \
                self._update_part_view(fv_vis, facv_shid, 
                                       self.vis[view], self.shid)
            dbias_shid += dbias_shid_view
            dweights_shrd.append(dweights_shrd_view)

        dbias_shid /= self.n_views

        return (dbias_vis, dbias_phid, dbias_shid,
                dweights_priv, dweights_shrd)

    def _update_part_view(self, fv_vis, facv_hid, distr_vis, distr_hid):
        # fv_vis[from_node, sample, from_statistic]
        # dlpv_hid[to_node, sample, to_statistic]
        dlpv_hid = distr_hid.dlp(facv_hid)

        dbias_vis = fv_vis.mean(axis=1)
        dbias_hid = dlpv_hid.mean(axis=1)

        dweights = T.zeros((dlpv_hid.shape[0], distr_hid.n_statistics, 
                            fv_vis.shape[0], distr_vis.n_statistics),
                           dtype=theano.config.floatX)
        for to_statistic in range(distr_hid.n_statistics):
            for from_statistic in range(distr_vis.n_statistics):
                dweights = T.set_subtensor(dweights[:, to_statistic, :, from_statistic],
                                           T.dot(dlpv_hid[:, :, to_statistic], 
                                                 fv_vis[:, :, from_statistic].T) /
                                           fv_vis.shape[1])

        return dbias_vis, dbias_hid, dweights

    def cd_learning_update(self, x_vis):
        """Parameter updates for contrastive divergence learning.
           Given the visibles x_vis[view][node] returns a tuple of updates for 
           the
           (visble biases, private latent biases, shared latent biases,
           private visible to private latent weights, 
           private visible to shared latent weights)
        """

        # data distribution        
        (data_dbias_vis, data_dbias_phid, data_dbias_shid,
         data_dweights_priv, data_dweights_shrd) = \
             self._update_part(x_vis)

        # model distribution
        model_x_vis = self.gibbs_sample_vis(x_vis, None, None, None, None, None,
                                            self.n_gs_learn)
        (model_dbias_vis, model_dbias_phid, model_dbias_shid,
         model_dweights_priv, model_dweights_shrd) = \
             self._update_part(model_x_vis)

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
