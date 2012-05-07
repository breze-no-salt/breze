# -*- coding: utf-8 -*-

import collections

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm


class RestrictedBoltzmannMachine(Model):

    def __init__(self, n_inpt, n_feature, inpt_dist='bernoulli', seed=1010):
        self.n_inpt = n_inpt
        self.n_feature = n_feature
        self.inpt_dist = inpt_dist
        # TODO check if it is a good idea to have it here; side effects?
        self.srng = RandomStreams(seed=seed)

        super(RestrictedBoltzmannMachine, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(self.n_inpt, self.n_feature)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        inpt = T.matrix('inpt')
        feature = T.matrix('feature')
        n_gibbs_steps = T.iscalar()
        self.exprs, self.updates = self.make_exprs(
            inpt, feature, self.parameters.in_to_feature,
             self.parameters.in_bias, self.parameters.feature_bias,
             n_gibbs_steps, self.srng)

    @staticmethod
    def get_parameter_spec(n_inpt, n_feature):
        return {'in_bias': n_inpt,
                'feature_bias': n_feature,
                'in_to_feature': (n_inpt, n_feature)}

    @staticmethod
    def make_exprs(inpt, feature, in_to_feature, in_bias, feature_bias, 
                   n_gibbs_steps, srng):
        # Energy of configuration.
        energy = ((inpt * in_bias.dimshuffle('x', 0)).sum(axis=1)
                  + (feature * feature_bias.dimshuffle('x', 0)).sum(axis=1)
                  + (T.dot(inpt, in_to_feature) * feature).sum(axis=1))
        import theano.printing
        energy = theano.printing.Print('energy')(energy)

        # Free energy given the visibles.
        free_energy_given_visibles = T.log((
            - T.dot(inpt, in_bias)
            - (1 + T.exp(T.dot(inpt, in_to_feature) + feature_bias)).sum(axis=1)))

        # Probability of features conditioned on visibles, p(h|v).
        p_feature_given_inpt = (
            transfer.sigmoid(T.dot(inpt, in_to_feature) + feature_bias))

        # Sample of features given visibles.
        feature_sample = (
            p_feature_given_inpt > srng.uniform(p_feature_given_inpt.shape))

        # Probability of visibles conditioned on features.
        p_inpt_given_feature = (
            transfer.sigmoid(T.dot(feature, in_to_feature.T) + in_bias))

        # Sample of visibles given features.
        inpt_sample = (p_inpt_given_feature > srng.uniform(inpt.shape))

        def gibbs_step(v):
            h = transfer.sigmoid(T.dot(v, in_to_feature) + feature_bias)
            sampled_h = h > srng.uniform(h.shape)
            recons = transfer.sigmoid(
                T.dot(sampled_h, in_to_feature.T) + in_bias)
            return recons

        gibbs_sample_visible, gibbs_sample_visible_updates= theano.scan(
            gibbs_step, outputs_info=[inpt], n_steps=n_gibbs_steps)
        gibbs_sample_visible = gibbs_sample_visible[-1]

        gibbs_sample_p_feature = transfer.sigmoid(
            T.dot(gibbs_sample_visible, in_to_feature) + feature_bias)
        gibbs_sample_p_feature_updates = gibbs_sample_visible_updates

        exprs = {
            'inpt': inpt,
            'feature': feature,
            'n_gibbs_steps': n_gibbs_steps,
            'energy': energy,
            'free_energy_given_visibles': free_energy_given_visibles,
            'p_feature_given_inpt': p_feature_given_inpt,
            'p_inpt_given_feature': p_inpt_given_feature,
            'feature_sample': feature_sample,
            'inpt_sample': inpt_sample,
            'gibbs_sample_visible': gibbs_sample_visible,
            'gibbs_sample_p_feature': gibbs_sample_p_feature,
        }

        updates = collections.defaultdict(lambda: {})
        updates.update({
            gibbs_sample_visible: gibbs_sample_visible_updates,
            gibbs_sample_p_feature: gibbs_sample_p_feature_updates
        })

        return exprs, updates
