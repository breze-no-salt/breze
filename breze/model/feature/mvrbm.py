# -*- coding: utf-8 -*-

import collections
import types

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm


class MultiViewHarmoniumModel:

    # Model hyperparameters:
    # f_vis[view](x_vis[node, sample]) -> fv_vis[node, sample, statistic]
    # f_phid[view](x_phid[node, sample]) -> fv_phid[node, sample, statistic]
    # f_shid(x_shid[node, sample]) -> fv_shid[node, sample, statistic]
    # lp_vis[view](fac[node, sample, statistic]) -> lpv_vis[node, sample]
    # lp_phid[view](fac[node, sample, statistic]) -> lpv_phid[node, sample]
    # lp_shid(fac[node, sample, statistic]) -> lpv_shid[node, sample]
    # dlp_phid[view](fac[node, sample, statistic]) -> dlpv_phid[node, sample, statistic]
    # dlp_shid(fac[node, sample, statistic]) -> dlpv_shid[node, sample, statistic]
    # sampler_vis[view](fac[node, sample, statistic]) -> sample_vis[node, sample]
    # sampler_phid[view](fac[node, sample, statistic]) -> sample_phid[node, sample]
    # sampler_shid(fac[node, sample, statistic]) -> sample_shid[node, sample]

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

    def __init__(self, n_views):
        # Hyperparameters
        self.f_vis = [None for _ in range(n_views)]
        self.f_phid = [None for _ in range(n_views)]
        self.f_shid = None
        self.lp_vis = [None for _ in range(n_views)]
        self.lp_phid = [None for _ in range(n_views)]
        self.lp_shid = None
        self.dlp_phid = [None for _ in range(n_views)]
        self.dlp_shid = None
        self.sampler_vis = [None for _ in range(n_views)]
        self.sampler_phid = [None for _ in range(n_views)]
        self.sampler_shid = None

        # Parameters
        self.bias_vis = [None for _ in range(n_views)]
        self.bias_phid = [None for _ in range(n_views)]
        self.bias_shid = None
        self.weights_priv = [None for _ in range(n_views)]
        self.weights_shrd = [None for _ in range(n_views)]

    def check_dimensions(self):
        n_views = len(self.f_vis)
        assert len(self.f_phid) == n_views
        assert len(self.bias_vis) == n_views
        assert len(self.bias_phid) == n_views
        assert len(self.weights_priv) == n_views
        assert len(self.weights_shrd) == n_views

    @property
    def n_views(self):
        return len(self.f_vis)

    @property
    def n_vis_statistics(self):
        return [len(f) for f in self.f_vis]

    @property
    def n_vis_nodes(self):
        return [f.shape[0] for f in self.bias_vis]
  
    @property
    def n_phid_statistics(self):
        return [len(f) for f in self.f_phid]

    @property
    def n_phid_nodes(self):
        return [f.shape[0] for f in self.bias_phid]

    @property
    def n_shid_statistics(self):
        return len(self.f_shid)

    @property
    def n_shid_nodes(self):
        return self.bias_shid.shape[0]

    def fac_shid(self, x_vis):
        # calculate factor of shared hidden units
        # fac_shid[node, sample, statistic]

        n_samples = x_vis[0].shape[1]
        facv_shid = np.zeros((self.n_shid_nodes, 
                              n_samples, 
                              self.n_shid_statistics))
        for statistic in range(self.n_shid_statistics):
            facv_shid[:, :, statistic] = np.tile(self.bias_shid[:, statistic],
                                                 (n_samples, 1)).T
            for from_view in range(n_views):
                fv_vis = self.f_vis[from_view](x_vis[from_view])
                for from_statistic in range(n_vis_statistics[from_view]):
                    facv_shid[:, :, statistic] += \
                        T.dot(self.weights_shrd[from_view][:, statistic, :, from_statistic].T,
                                                fv_vis[:, :, from_statistic])
        return facv_shid
            
    def p_shid(self, x_shid, x_vis):
        """Probability p_shid[node, sample] of shared hidden units having values 
            x_shid[node, sample] given that visible units have values 
            x_vis[view][node, sample]"""
        # p_shid[node, sample]
        fv_shid = self.f_shid(x_shid)
        facv_shid = self.fac_shid(x_vis)
        lpv_shid = self.lp_shid(facv_shid)
        return (facv_shid * fv_shid).sum(axis=2) - lpv_shid

    def sample_shid(self, x_vis):
        """Sample shared hidden units x_shid[node, sample] given that visible units 
            have values x_vis[view][node, sample]"""
        facv_shid = self.fac_shid(x_vis)
        return self.sampler_shid(facv_shid)

    def fac_phid(self, x_vis):
        # calculate probability of private hidden units
        # fac_phid[view][node, sample, statistic]
        n_samples = x_vis[0].shape[1]
        facv_phid = [np.zeros((self.n_phid_nodes[view],
                               n_samples,
                               self.n_phid_statistics[view])) 
                     for view in range(self.n_views)]
        for view in range(self.n_views):      
            fv_vis = self.f_vis[view](x_vis[view])
            for statistic in range(self.n_phid_statistics[view]):
                facv_phid[view][:, :, statistic] = np.tile(self.bias_phid[view][:, statistic],
                                                           (n_samples, 1)).T
                for from_statistic in range(n_vis_statistics[view]):
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
            fv_phid = self.f_phid[view](x_phid[view])
            lpv_phid = self.lp_phid[view](facv_phid[view])
            pv_phid.append((facv_phid[view] * fv_phid).sum(axis=2) - lpv_phid)
        return pv_phid

    def sample_phid(self, x_vis):
        """Sample private hidden units x_phid[view][node, sample] given that 
            visible units have values x_vis[view][node, sample]"""
        facv_phid = self.fac_phid(x_vis)
        samplev_phid = []
        for view in range(self.n_views):
            samplev_phid.append(self.sampler_phid[view](facv_phid[view]))
        return samplev_phid

    def fac_vis(self, x_phid, x_shid):
        # calculate probability of visible units
        # fac_vis[view][node, sample, statistic]
        n_samples = x_vis[0].shape[1]
        facv_vis = [np.zeros((self.n_vis_nodes[view],
                              n_samples,
                              self.n_vis_statistics[view])) 
                    for view in range(self.n_views)]
        fv_shid = self.f_shid(x_shid)
        for view in range(self.n_views):      
            fv_phid = self.f_phid[view](x_phid[view])
            for statistic in range(self.n_vis_statistics[view]):
                facv_vis[view][:, :, statistic] = np.tile(self.bias_vis[view][:, statistic],
                                                          (n_sample, 1)).T
                for from_statistic in range(self.n_phid_statistics[view]):
                    facv_vis[view][:, :, statistic] += \
                        T.dot(weights_priv[view][:, statistic, :, from_statistic],
                              fv_phid[:, :, from_statistic])
                for from_statistic in range(self.n_shid_statistics):
                    facv_vis[view][:, :, statistic] += \
                        T.dot(weights_shrd[view][:, statistic, :, from_statistic],
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
            fv_vis = self.f_vis[view](x_vis[view])
            lpv_vis = self.lp_vis[view](facv_vis[view])
            pv_vis.append((facv_vis[view] * fv_vis).sum(axis=2) - lpv_vis)
        return pv_vis

    def sample_vis(self, x_phid, x_shid):
        """Sample visible units x_vis[view][node, sample] given that private 
            hidden units have values x_phid[view][node, sample] and shared hidden 
            units have values x_shid[node, sample]"""
        facv_vis = self.fac_vis(x_phid, x_shid)
        samplev_vis = []
        for view in range(self.n_views):
            samplev_vis.append(self.sampler_vis[view](facv_vis[view]))
        return samplev_vis

    def gibbs_sample_vis(self, x_vis_start, x_phid_start, x_shid_start,
                         vis_clamp, phid_clamp, shid_clamp,
                         n_iterations):
        n_samples = x_vis_start.shape[1]
        x_vis = x_vis_start
        for i in range(n_iterations):
            # sample private and shared hiddens given visibles
            x_phid = self.sample_phid(x_vis)
            if phid_clamp is not None:
                for view in range(n_views):
                    if phid_clamp[view]:
                        x_phid[view] = x_phid_start[view]
            if not shid_clamp:
                x_shid = self.sample_shid(x_vis)
            else:
                x_shid = x_shid_start

            # sample visibles given hiddens
            x_vis = self.sample_vis(x_phid, x_shid)
            if vis_clamp is not None:
                for view in range(n_views):
                    if vis_clamp[view]:
                        x_vis = x_vis_start[view]

        return x_vis

    def _update_part(self, x_vis, fac, dlp, n_hid_statistics):
        to_multiview = isinstance(fac, collections.Iterable)
        if not to_multiview:
            facv = fac(x_vis)
            dlpv = dlp(facv)
            n_hid_statistics_view = n_hid_statistics

        # fv_vis[from_node, sample, from_statistic]
        # dlpv[to_node, sample, to_statistic]
        fv_vis = [self.f_vis(x) for x in x_vis]
        dbias_vis = []
        for view in range(self.n_views):
            if to_multiview:
                facv = fac[view](x_vis)
                dlpv = dlp[view](facv)
                n_hid_statistics_view = n_hid_statistics[view]

            dbias_vis.append(fv_vis.sum(axis=1))
            dbias_hid.append(dlpv.sum(axis=1))

            for to_statistic in range(n_hid_statistics_view):
                for from_statistic in range(self.n_vis_statistics[view]):
                    dweights[view][:, to_statistic, : from_statistic] = \
                        T.dot(dlpv[:, :, to_statistic], fv_vis[:, :, from_statistic].T)

        return (dbias_vis, dbias_hid, dweights)


    def cd_learning_update(self, x_vis, n_gibbs_iterations):

        # data distribution        
        (data_dbias_vis, data_dbias_phid, data_dweights_priv) = \
            self._update_part(x_vis, self.fac_phid, self.dlp_phid, 
                              self.n_phd_statistics)
        (_, data_dbias_shid, data_dweights_shrd) = \
            self._update_part(x_vis, self.fac_shid, self.dlp_shid, 
                              self.n_shid_statistics)

        # model distribution
        model_x_vis = self.gibbs_sample_vis(x_vis, None, None, None, None, None,
                                            n_gibbs_iterations)
        (model_dbias_vis, model_dbias_phid, model_dweights_priv) = \
            self._update_part(model_x_vis, self.fac_phid, self.dlp_phid, 
                              self.n_phd_statistics)
        (_, model_dbias_shid, model_dweights_shrd) = \
            self._update_part(x_vis, self.fac_shid, self.dlp_shid, 
                              self.n_shid_statistics)

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


class MultiViewHarmonium(Model):

    def __init__(self, n_views, n_vis, n_phid, n_shid, 
                 n_vis_statistics, n_phid_statistics, n_shid_statistics,
                 n_gs_learn,
                 f_vis, f_phid, f_shid,
                 lp_vis, lp_phid, lp_shid,
                 dlp_phid, dlp_shid,
                 sampler_vis, sampler_phid, sampler_shid,
                 seed=1010):
        self.n_views = n_views
        self.n_vis = n_vis
        self.n_phid = n_phid
        self.n_shid = n_shid
        self.n_vis_statistics = n_vis_statistics
        self.n_phid_statistics = n_phid_statistics
        self.n_shid_statistics = n_shid_statistics
        self.n_gs_learn = n_gs_learn

        self.model = MultiViewHarmoniumModel(n_views)
        self.model.f_vis = f_vis
        self.model.f_phid = f_phid
        self.model.f_shid = f_shid
        self.model.lp_vis = lp_vis
        self.model.lp_phid = lp_phid
        self.model.lp_shid = lp_shid
        self.model.dlp_phid = dlp_phid
        self.model.dlp_shid = dlp_shid
        self.model.sampler_vis = sampler_vis
        self.model.sampler_phid = sampler_phid
        self.model.sampler_shid = sampler_shid

        self.srng = RandomStreams(seed=seed)

        super(MultiViewHarmonium, self).__init__()

    def init_pars(self):       
        parspec = self.get_parameter_spec(self.n_views,
                                          self.n_vis, self.n_phid, self.n_shid,
                                          self.n_vis_statistics, 
                                          self.n_phid_statistics, 
                                          self.n_shid_statistics)
        self.parameters = ParameterSet(**parspec)

        self.model.bias_shid = self.parameters['bias_shid']
        for view in range(self.n_views):
            self.model.bias_vis[view] = self.parameters['bias_vis[' + view + ']']
            self.model.bias_phid[view] = self.parameters['bias_phid[' + view + ']']
            self.model.weights_priv[view] = self.parameters['weights_priv[' + view + ']']
            self.model.weights_shrd[view] = self.parameters['weights_shrd[' + view + ']']

    @staticmethod
    def get_parameter_spec(n_views,
                           n_vis, n_phid, n_shid,
                           n_vis_statistics, n_phid_statistics, n_shid_statistics):
        ps = {'bias_shid': (n_shid, n_shid_statistics)}
        for view in range(n_views):
            ps['bias_vis[' + view + ']'] = (n_vis[view], n_vis_statistics[view])
            ps['bias_phid[' + view + ']'] = (n_phid[view], n_phid_statistics[view])
            ps['weights_priv[' + view + ']'] = (n_phid[view], n_phid_statistics[view],
                                                n_vis[view], n_vis_statistics[view])
            ps['weights_shrd[' + view + ']'] = (n_shid, n_shid_statistics,
                                                n_vis[view], n_vis_statistics[view])
        return ps

    def init_exprs(self):
        x_vis = [T.matrix('x_vis[' + view + ']') for view in range(self.n_views)]
        x_phid = [T.matrix('x_phid[' + view + ']') for view in range(self.n_views)]
        x_shid = T.matrix('x_shid')

        bias_vis = [self.parameters['bias_vis[' + view +']'] 
                    for view in range(self.n_views)]
        bias_phid = [self.parameters['bias_phid[' + view + ']']
                     for view in range(self.n_views)]
        weights_priv = [self.parameters['weights_priv[' + view + ']']
                        for view in range(self.n_views)]
        weights_shrd = [self.parameters['weights_shrd[' + view + ']']
                        for view in range(self.n_views)]

        self.exprs, self.updates = self.make_exprs(self.model,
                                                   x_vis, x_phid, x_shid,
                                                   bias_vis, bias_phid, bias_shid,
                                                   weights_priv, weights_shrd,
                                                   self.n_gs_learn,
                                                   self.srng)
    @staticmethod
    def make_exprs(model,
                   x_vis, x_phid, x_shid,
                   bias_vis, bias_phid, bias_shid,
                   weights_priv, weights_shrd,
                   n_gs_learn,
                   srng):

        def p(q, val, dist):
            if dist == 'bernoulli':
                if val == 1:
                    return transfer.sigmoid(q)
                elif val == 0:
                    return 1 - transfer.sigmoid(q)
                else:
                    assert False
            elif dist == 'gaussian':
                return 1/T.sqrt(2*pi) * T.exp(-1/2 * T.sqr(val-q))
            elif dist == 'relu':
                if val == 0:
                    TODO
                else:
                    return 1/T.sqrt(2*pi) * T.exp(-1/2 * T.sqr(val-q))

        def features(x, y):                    
            # p(h_x|x)
            p_x_feature = transfer.sigmoid(T.dot(x, x_to_x_feature) + x_feature_bias)
            x_feature_sample = p_x_feature > srng.uniform(p_x_feature.shape)

            # p(h_y|y)
            p_y_feature = transfer.sigmoid(T.dot(x, y_to_y_feature) + y_feature_bias)
            y_feature_sample = p_y_feature > srng.uniform(p_y_feature.shape)

            # p(h_c|x,y)
            p_common_feature = transfer.sigmoid(T.dot(x, x_to_common_feature) + 
                                                T.dot(y, y_to_common_feature) +
                                                common_feature_bias)
            common_feature_sample = (p_common_feature > 
                                     srng.uniform(p_common_feature.shape))

            return p_x_feature, x_feature_sample, p_y_feature, y_feature_sample, \
                p_common_feature, common_feature_sample

        def visibles(x_feature, y_feature, common_feature):
            # p(x|h_x,h_c)
            p_x = transfer.sigmoid(T.dot(x_feature, x_to_x_feature.T) + 
                                   T.dot(common_feature, x_to_common_feature.T) +
                                   x_bias)
            x_sample = p_x > srng.uniform(p_x.shape)

            # p(y|h_y,h_c)
            p_y = transfer.sigmoid(T.dot(y_feature, y_to_y_feature.T) + 
                                   T.dot(common_feature, y_to_common_feature.T) +
                                   y_bias)
            y_sample = p_y > srng.uniform(p_y.shape)

            return p_x, x_sample, p_y, y_sample

        def gibbs_step(xy_start, clamp=[]):
            # does one iteration of gibbs sampling
            [x_start, y_start] = xy_start
            _, x_feature_sample, _, y_feature_sample, _, common_feature_sample = \
                features(x_start, y_start)
            if 'x_feature' in clamp:
                x_feature_sample = x_feature
            if 'y_feature' in clamp:
                y_feature_sample = y_feature
            if 'common_feature' in clamp:
                common_feature_sample = common_feature          
            p_x_recon, _, p_y_recon, _ = visibles(x_feature_sample,
                                                  y_feature_sample,
                                                  common_feature_sample)
            if 'x' in clamp:
                p_x_recon = x_start
            if 'y' in clamp:
                p_y_recon = y_start
            return [p_x_recon, p_y_recon]

        # features given visibles
        p_x_feature, x_feature_sample, p_y_feature, y_feature_sample, \
            p_common_feature, common_feature_sample = features(x, y)

        # visibles given features
        p_x, x_sample, p_y, y_sample = visibles(x_feature, y_feature, common_feature)

        # gibbs sampling for learning
        gs_p_xy, gs_p_xy_updates = theano.scan(lambda inpt: gibbs_step(input, []), 
                                               outputs_info=[x,y], 
                                               n_steps=n_gs_learn)
        gs_p_xy = gs_p_xy[-1]
        [gs_p_x, gs_p_y] = gs_p_xy
        gs_p_x_feature, _, gs_p_y_feature, _, gs_p_common_feature, _ = \
            features(gs_p_x, gs_p_y)

        # gibbs sampling for inference of x from (y,h_x)
        infer_p_x_with_x_feature, infer_p_x_with_x_feature_updates = \
            theano.scan(lambda inpt: gibbs_step(inpt, ['y', 'x_feature']),
                        outputs_info=[x,y], 
                        n_steps=n_gs_infer)
        infer_p_x_with_x_feature = infer_p_x_with_x_feature[-1]
        [infer_p_x_with_x_feature, _] = infer_p_x_with_x_feature

        exprs = {
            'x': x,
            'y': y,
            'x_feature': x_feature,
            'y_feature': y_feature,
            'common_feature': common_feature,
            'n_gs_learn': n_gs_learn,
            'n_gs_infer': n_gs_infer,
            'p_x_feature': p_x_feature,
            'x_feature_sample': x_feature_sample,
            'p_y_feature': p_y_feature,
            'y_feature_sample': y_feature_sample,            
            'p_common_feature': p_common_feature,
            'common_feature_sample': common_feature_sample,
            'p_x': p_x,
            'x_sample': x_sample,
            'p_y': p_y,
            'y_sample': y_sample,
            'gs_p_x': gs_p_x,
            'gs_p_y': gs_p_y,
            'gs_p_x_feature': gs_p_x_feature,
            'gs_p_y_feature': gs_p_y_feature,
            'gs_p_common_feature': gs_p_common_feature,
            'infer_p_x_with_x_feature': infer_p_x_with_x_feature,
        }

        updates = collections.defaultdict(lambda: {})
        updates.update({
            gs_p_x: gs_p_xy_updates,
            gs_p_y: gs_p_xy_updates,
            gs_p_x_feature: gs_p_xy_updates,
            gs_p_y_feature: gs_p_xy_updates,
            gs_p_common_feature: gs_p_xy_updates,
            infer_p_x_with_x_feature: infer_p_x_with_x_feature_updates,
        })

        return exprs, updates
