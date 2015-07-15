# -*- coding: utf-8 -*-


import climin.initialize
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet

from theano.compile import optdb

from breze.arch.construct.layer.distributions import NormalGauss
from breze.arch.construct.layer.varprop.sequential import FDRecurrent
from breze.arch.construct.layer.varprop.simple import AffineNonlinear
from breze.arch.construct.neural import distributions as neural_dists
from breze.learn.utils import theano_floatx

from base import GenericVariationalAutoEncoder


class BernoulliVisibleStornMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.FastDropoutRnnBernoulli(
            latent_sample,
            n_inpt=self.n_latent + self.n_inpt,
            n_hiddens=self.n_hiddens_gen,
            n_output=self.n_inpt,
            hidden_transfers=self.gen_transfers,
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            p_dropout_hidden_to_out=self.p_dropout_hidden_to_out,
            declare=self.parameters.declare)


class GaussVisibleStornMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.FastDropoutRnnDiagGauss(
            latent_sample,
            n_inpt=self.n_latent + self.n_inpt,
            n_hiddens=self.n_hiddens_gen,
            n_output=self.n_inpt,
            hidden_transfers=self.gen_transfers,
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            p_dropout_hidden_to_out=self.p_dropout_hidden_to_out,
            declare=self.parameters.declare)


class ConstVarGaussVisibleStornMixin(object):

    shared_std = False
    fixed_std = None

    def make_gen(self, latent_sample):
        return neural_dists.FastDropoutRnnConstDiagGauss(
            latent_sample,
            n_inpt=self.n_latent + self.n_inpt,
            n_hiddens=self.n_hiddens_gen,
            n_output=self.n_inpt,
            hidden_transfers=self.gen_transfers,
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            p_dropout_hidden_to_out=self.p_dropout_hidden_to_out,
            shared_std=self.shared_std,
            fixed_std=self.fixed_std,
            declare=self.parameters.declare)


class GaussLatentStornMixin(object):

    distribution_klass = neural_dists.FastDropoutRnnDiagGauss

    def make_prior(self, sample):
        return NormalGauss(sample.shape)

    def make_recog(self, inpt):
        return self.distribution_klass(
            inpt,
            n_inpt=self.n_inpt,
            n_hiddens=self.n_hiddens_recog,
            n_output=self.n_latent,
            hidden_transfers=self.recog_transfers,
            p_dropout_inpt='parameterized',
            p_dropout_hiddens=['parameterized' for _ in self.n_hiddens_recog],
            p_dropout_hidden_to_out='parameterized',
            declare=self.parameters.declare)


class GaussLatentBiStornMixin(GaussLatentStornMixin):

    distribution_klass = neural_dists.FastDropoutBiRnnDiagGauss


class StochasticRnn(GenericVariationalAutoEncoder):

    sample_dim = 1,
    theano_optimizer = optdb.query(theano.gof.Query(
        include=['fast_run'], exclude=['scan_eqopt1', 'scan_eqopt2']))
    mode = theano.Mode(linker='cvm', optimizer=theano_optimizer)

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 p_dropout_inpt=.1, p_dropout_hiddens=.1,
                 p_dropout_hidden_to_out=None,
                 p_dropout_shortcut=None,
                 use_imp_weight=False,
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):

        self.n_hiddens_recog = n_hiddens_recog
        self.n_hiddens_gen = n_hiddens_gen

        self.recog_transfers = recog_transfers
        self.gen_transfers = gen_transfers

        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        self.p_dropout_hidden_to_out = p_dropout_hidden_to_out
        self.p_dropout_shortcut = p_dropout_shortcut

        super(StochasticRnn, self).__init__(
            n_inpt, n_latent,
            use_imp_weight=use_imp_weight,
            batch_size=batch_size, optimizer=optimizer,
            max_iter=verbose, verbose=verbose)

    def _make_start_exprs(self):
        inpt = T.tensor3('inpt')
        inpt.tag.test_value, = theano_floatx(np.ones((4, 3, self.n_inpt)))

        if self.use_imp_weight:
            imp_weight = T.tensor3('imp_weight')
            imp_weight.tag.test_value, = theano_floatx(np.ones((4, 3, 1)))
        else:
            imp_weight = None

        return inpt, imp_weight

    def make_cond(self, inpt):
        return T.concatenate([T.zeros_like(inpt[:1]), inpt[:-1]], 0)

    def initialize(self,
                   par_std=1, par_std_affine=None, par_std_rec=None,
                   par_std_in=None,
                   sparsify_affine=None, sparsify_rec=None,
                   spectral_radius=None):
        climin.initialize.randomize_normal(self.parameters.data, 0, par_std)
        all_layers = self.vae.recog.rnn.layers + self.vae.gen.rnn.layers
        P = self.parameters
        for i, layer in enumerate(all_layers):
            if isinstance(layer, FDRecurrent):
                p = P[layer.weights]
                if par_std_rec:
                    climin.initialize.randomize_normal(p, 0, par_std_rec)
                if sparsify_rec:
                    climin.initialize.sparsify_columns(p, sparsify_rec)
                if spectral_radius:
                    climin.initialize.bound_spectral_radius(p, spectral_radius)
                P[layer.initial_mean][...] = 0
                P[layer.initial_std][...] = 1
            if isinstance(layer, AffineNonlinear):
                p = self.parameters[layer.weights]
                if par_std_affine:
                    if i == 0 and par_std_in:
                        climin.initialize.randomize_normal(p, 0, par_std_in)
                    else:
                        climin.initialize.randomize_normal(p, 0, par_std_affine)
                if sparsify_affine:
                    climin.initialize.sparsify_columns(p, sparsify_affine)

                self.parameters[layer.bias][...] = 0

    #def _make_gen_hidden(self):
    #    hidden_exprs = [T.concatenate(i.recurrent.outputs, 2)
    #                    for i in self.vae.gen.hidden_layers]

    #    return self.function(['inpt'], hidden_exprs)

    #def gen_hidden(self, X):
    #    if getattr(self, 'f_gen_hiddens', None) is None:
    #        self.f_gen_hiddens = self._make_gen_hidden()
    #    return self.f_gen_hiddens(X)

    def sample(self, n_time_steps, prefix=None, visible_map=False):
        if prefix is None:
            raise ValueError('need to give prefix')

        if not hasattr(self, 'f_gen'):
            vis_sample = self.vae.gen.inpt

            inpt_m1 = T.tensor3('inpt_m1')
            inpt_m1.tag.test_value = np.zeros((3, 2, self.n_inpt))

            latent_prior_sample = T.tensor3('latent_prior_sample')
            latent_prior_sample.tag.test_value = np.zeros((3, 2, self.n_latent))

            gen_inpt_sub = T.concatenate([latent_prior_sample, inpt_m1], axis=2)

            vis_sample = theano.clone(
                vis_sample,
                {self.vae.gen.inpt: gen_inpt_sub}
            )

            gen_out_sub = theano.clone(
                self.vae.gen.rnn.output, {self.vae.gen.inpt: gen_inpt_sub})
            self._f_gen_output = self.function(
                [inpt_m1, latent_prior_sample],
                gen_out_sub, mode='FAST_COMPILE',
                on_unused_input ='warn')

            out = self.vae.gen.sample() if not visible_map else self.vae.gen.maximum
            self._f_visible_sample_by_gen_output = self.function(
                [self.vae.gen.rnn.output], out,
                on_unused_input ='warn')

            def f_gen(inpt_m1, latent_prior_sample):
                rnn_out = self._f_gen_output(inpt_m1, latent_prior_sample)
                return self._f_visible_sample_by_gen_output(rnn_out)

            self.f_gen = f_gen

        prefix_length = prefix.shape[0]
        S = np.empty(
            (prefix.shape[0] + n_time_steps, prefix.shape[1],
             prefix.shape[2])
        ).astype(theano.config.floatX)
        S[:prefix_length][...] = prefix
        latent_samples = np.zeros(
            (prefix.shape[0] + n_time_steps, prefix.shape[1], self.n_latent)
            ).astype(theano.config.floatX)
        latent_samples[prefix_length:] = np.random.standard_normal(
            (n_time_steps, prefix.shape[1], self.n_latent))
        for i in range(n_time_steps):
            p = self.f_gen(
                S[:prefix_length + i], latent_samples[:prefix_length + i]
            )[-1, :, :self.n_inpt]
            S[prefix_length + i] = p

        return S[prefix_length + 1:]
