# -*- coding: utf-8 -*-

import collections

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm


class MultiViewRestrictedBoltzmannMachine(Model):

    def __init__(self, n_x, n_y, n_x_feature, n_y_feature, n_common_feature,
                 inpt_dist='bernoulli', seed=1010):
        self.n_x = n_x
        self.n_y = n_y
        self.n_x_feature = n_x_feature
        self.n_y_feature = n_y_feature
        self.n_common_feature = n_common_feature
        self.inpt_dist = inpt_dist
        # TODO check if it is a good idea to have it here; side effects?
        self.srng = RandomStreams(seed=seed)

        super(MultiViewRestrictedBoltzmannMachine, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(self.n_x, self.n_y, 
                                          self.n_x_feature, self.n_y_feature,
                                          self.n_common_feature)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        x = T.matrix('x')
        y = T.matrix('y')
        x_feature = T.matrix('x_feature')
        y_feature = T.matrix('y_feature')
        common_feature = T.matrix('common_feature')
        n_gs_learn = T.iscalar()
        n_gs_infer = T.iscalar()
        self.exprs, self.updates = self.make_exprs(
            x, y, x_feature, y_feature, common_feature,
            self.parameters.x_bias, self.parameters.y_bias,
            self.parameters.x_feature_bias, self.parameters.y_feature_bias, 
            self.parameters.common_feature_bias,
            self.parameters.x_to_x_feature, self.parameters.y_to_y_feature,
            self.parameters.x_to_common_feature, self.parameters.y_to_common_feature,
            n_gs_learn, n_gs_infer, self.srng)

    @staticmethod
    def get_parameter_spec(n_x, n_y, n_x_feature, n_y_feature, n_common_feature):
        return {'x_bias': n_x,
                'y_bias': n_y,
                'x_feature_bias': n_x_feature,
                'y_feature_bias': n_y_feature,
                'common_feature_bias': n_common_feature,
                'x_to_x_feature': (n_x, n_x_feature),
                'y_to_y_feature': (n_y, n_y_feature),
                'x_to_common_feature': (n_x, n_common_feature),
                'y_to_common_feature': (n_y, n_common_feature)}

    @staticmethod
    def make_exprs(x, y, x_feature, y_feature, common_feature, 
                   x_bias, y_bias,
                   x_feature_bias, y_feature_bias, common_feature_bias,
                   x_to_x_feature, y_to_y_feature, 
                   x_to_common_feature, y_to_common_feature,
                   n_gs_learn, n_gs_infer, 
                   xy_dist, feature_dist, srng):
        pass

    def mwh(f_vis, bias_vis, lp_vis, 
            f_phid, bias_phid, lp_phid, weights_priv,
            f_shid, fac_shid, lp_shid):

        # f_vis[view][statistic]
        # f_phid[view][statistic]
        # bias_vis[view][node, statistic]
        # bias_phid[view][node, statistic]
        # weights_priv[view][to_node, to_statistic, from_node, from_statistic]
        # weights_comm[view][to_node, to_statistic, from_node, from_statistic]
        
        n_views = len(f_vis)
        assert len(fac_vis) == n_views
        assert len(lp_vis) == n_views

        for view in range(n_views):
            for statistic in range(n_statistics):
                fac_vis[view][statistic] = bias_vis[i][:, statistic]
                for from_statistic in range(n_statistics):
                    fac_vis[view][statistic] += \
                        T.dot(weights_priv[view][:, statistic, :, from_statistic],
                              f_phid[view][from_statistic])

        p_vis[i] = (f_vis[i] * fac_vis[i]).sum(axis=1) - lp_vis[i]





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
