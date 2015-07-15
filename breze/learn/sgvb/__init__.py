# -*- coding: utf-8 -*-

"""Module for learning models with stochastic gradient variational Bayes (SGVB).

The method has been introduced in parallel by [SGVB]_ and [DLGM]_. The general
idea is to optimize a variational upper bound on the negative log-likelihood of
the data with stochastic gradient descent. The user is urged to review these
papers before using the models.


Methodological review
---------------------

We will give a short review for notation here. We consider the problem of
estimating a model for the data density :math:`p(x)`, where we assume it to be
driven by a set of latent variables :math:`z`. The data negative log likelihood
can then be bounded from above

.. math::
   -\\log p(x) \\le {\\text{KL}}(q(z|x)|p(z)) - \mathbb{E}_{z \\sim q}[\\log p(x|z)].

We will refer to :math:`q(z)` as the recognition model or approxiamte posterior.
:math:`p(x|z)` is the generating model. :math:`p(z)` is the prior over the
latent variables.
The first term is the Kullback-Leibler divergence between the approximate
posterior and the prior.
The second term is the expected negative log-likelihood of the data given the
recognition model.

Training of the model is performed by stochastic gradient descent. In practice,
we will select a mini batch of the data for which we can obtain :math:`q(z|x)`.
This can directly be used to calculate the first term. By sampling from that
model and putting it through the generating model we can approximate the second
term.


References
----------

.. [SGVB] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
.. [DLGM] Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic Back-propagation and Variational Inference in Deep Latent Gaussian Models." arXiv preprint arXiv:1401.4082 (2014).
"""


import climin.initialize
from climin import mathadapt as ma

import numpy as np

from scipy.misc import logsumexp

import theano
import theano.tensor as T
import theano.tensor.nnet
from theano.compile import optdb

from breze.arch.construct.layer.varprop.simple import AffineNonlinear
from breze.arch.construct.layer.varprop.sequential import FDRecurrent
from breze.arch.construct import neural
from breze.arch.construct.layer.distributions import NormalGauss
from breze.arch.construct.neural import distributions as neural_dists
from breze.arch.util import wild_reshape
from breze.learn.utils import theano_floatx

from base import GenericVariationalAutoEncoder


def flatten_list(lst):
    res = []
    for i in lst:
        res += list(i)
    return res


def assert_no_time(X):
    if X.ndim == 2:
        return X
    if X.ndim != 3:
        raise ValueError('ndim must be 2 or 3, but it is %i' % X.ndim)
    return wild_reshape(X, (-1, X.shape[2]))


def recover_time(X, time_steps):
    return wild_reshape(X, (time_steps, -1, X.shape[1]))


def normal_logpdf(xs, means, vrs):
    energy = -(xs - means) ** 2 / (2 * vrs)
    partition_func = -T.log(T.sqrt(2 * np.pi * vrs))
    return partition_func + energy


# TODO document
def estimate_nll(X, f_nll_z, f_nll_x_given_z, f_nll_z_given_x,
                 f_sample_z_given_x, n_samples):
    if X.ndim == 2:
        log_prior = np.empty((n_samples, X.shape[0]))
        log_posterior = np.empty((n_samples, X.shape[0]))
        log_recog = np.empty((n_samples, X.shape[0]))
    elif X.ndim == 3:
        log_prior = np.empty((n_samples, X.shape[0], X.shape[1]))
        log_posterior = np.empty((n_samples, X.shape[0], X.shape[1]))
        log_recog = np.empty((n_samples, X.shape[0], X.shape[1]))
    else:
        raise ValueError('unexpected ndim for X, can be 2 or 3')

    for i in range(n_samples):
        Z = f_sample_z_given_x(X)

        log_prior[i] = ma.assert_numpy(-f_nll_z(Z))
        log_posterior[i] = ma.assert_numpy(-f_nll_x_given_z(X, Z))
        log_recog[i] = ma.assert_numpy(-f_nll_z_given_x(Z, X))

    d = log_prior + log_posterior - log_recog

    while d.ndim > 1:
        d = d.sum(-1)
    ll = logsumexp(d, 0) - np.log(n_samples)

    # Normalize to average.
    ll /= X.shape[0]
    if X.ndim == 3:
        ll /= X.shape[1]
    return -ll


class MlpGaussLatentVAEMixin(object):

    def make_prior(self, sample):
        return NormalGauss(sample.shape)

    def make_recog(self, inpt):
        return  neural_dists.MlpDiagGauss(
            inpt, self.n_inpt,
            self.n_hiddens_recog,
            self.n_latent,
            self.recog_transfers,
            out_transfer_mean='identity',
            out_transfer_var=lambda x: x ** 2 + 1e-5,
            declare=self.parameters.declare)


class MlpGaussVisibleVAEMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.MlpDiagGauss(
            latent_sample, self.n_latent,
            self.n_hiddens_gen,
            self.n_inpt,
            self.gen_transfers,
            # TODO where to get the transfers from?
            # remove lambda
            out_transfer_mean='identity',
            out_transfer_var=lambda x: x ** 2 + 1e-5,
            declare=self.parameters.declare)


class MlpBernoulliVisibleVAEMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.MlpBernoulli(
            latent_sample, self.n_latent,
            self.n_hiddens_gen,
            self.n_inpt,
            self.gen_transfers,
            declare=self.parameters.declare)


class FastDropoutMlpGaussLatentVAEMixin(object):

    def make_prior(self, sample):
        return NormalGauss(sample.shape)

    def make_recog(self, inpt):
        return neural_dists.FastDropoutMlpDiagGauss(
            inpt, self.n_inpt,
            self.n_hiddens_recog,
            self.n_latent,
            self.recog_transfers,
            out_transfer='identity',
            dropout_parameterized=True,
            declare=self.parameters.declare)


class FastDropoutMlpGaussVisibleVAEMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.FastDropoutMlpDiagGauss(
            latent_sample, self.n_latent,
            self.n_hiddens_gen,
            self.n_inpt,
            self.gen_transfers,
            out_transfer='identity',
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            declare=self.parameters.declare)


class FastDropoutMlpBernoulliVisibleVAEMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.FastDropoutMlpBernoulli(
            latent_sample, self.n_latent,
            self.n_hiddens_gen,
            self.n_inpt,
            self.gen_transfers,
            out_transfer='sigmoid',
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            declare=self.parameters.declare)


class VariationalAutoEncoder(GenericVariationalAutoEncoder):

    """
        Original Variational Autoencoder with Mlp as recognition and generative model
    """

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 use_imp_weight=False,
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):
        self.n_hiddens_recog = n_hiddens_recog
        self.n_hiddens_gen = n_hiddens_gen
        self.recog_transfers = recog_transfers
        self.gen_transfers = gen_transfers

        super(VariationalAutoEncoder, self).__init__(
            n_inpt, n_latent,
            use_imp_weight=use_imp_weight,
            batch_size=batch_size, optimizer=optimizer,
            max_iter=verbose, verbose=verbose)


class FastDropoutVariationalAutoEncoder(VariationalAutoEncoder):

    """
        Variational Autoencoder with FastDropout Mlp as recognition and generative model
    """

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 p_dropout_inpt=.1, p_dropout_hiddens=.1,
                 use_imp_weight=False,
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens

        super(FastDropoutVariationalAutoEncoder, self).__init__(
            n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
            recog_transfers, gen_transfers,
            use_imp_weight=use_imp_weight,
            batch_size=batch_size, optimizer=optimizer,
            max_iter=verbose, verbose=verbose)


class ConvolutionalVAE(GenericVariationalAutoEncoder):

    def __init__(self, image_height, image_width, n_channel,
                 recog_n_hiddens_conv,
                 recog_filter_shapes, recog_pool_shapes,
                 recog_n_hiddens_full,
                 recog_transfers_conv, recog_transfers_full,
                 n_latent,
                 gen_n_hiddens_full,
                 gen_n_hiddens_conv,
                 gen_filter_shapes, gen_unpool_factors,
                 gen_transfers_conv, gen_transfers_full,
                 assumptions,
                 recog_strides=None,
                 use_imp_weight=False,
                 batch_size=None,
                 optimizer='adam',
                 max_iter=1000, verbose=False):
        self.image_height = image_height
        self.image_width = image_width
        self.n_channel = n_channel

        self.recog_n_hiddens_conv = recog_n_hiddens_conv
        self.recog_filter_shapes = recog_filter_shapes
        self.recog_pool_shapes = recog_pool_shapes
        self.recog_n_hiddens_full = recog_n_hiddens_full
        self.recog_transfers_conv = recog_transfers_conv
        self.recog_transfers_full = recog_transfers_full
        self.recog_strides = recog_strides

        self.n_latent = n_latent
        self.gen_n_hiddens_conv = gen_n_hiddens_conv
        self.gen_filter_shapes = gen_filter_shapes
        self.gen_unpool_factors = gen_unpool_factors
        self.gen_n_hiddens_full = gen_n_hiddens_full
        self.gen_transfers_conv = gen_transfers_conv
        self.gen_transfers_full = gen_transfers_full

        rec_class = lambda inpt, declare: neural.Lenet(
            inpt, self.image_height, self.image_width, self.n_channel,
            self.recog_n_hiddens_conv,
            self.recog_filter_shapes, self.recog_pool_shapes,
            self.recog_n_hiddens_full,
            self.recog_transfers_conv, self.recog_transfers_full,
            assumptions.latent_layer_size(self.n_latent),
            assumptions.statify_latent,
            strides=self.recog_strides,
            declare=declare)

        gen_class = lambda inpt, declare: neural.DeconvNet2d(
            inpt=inpt, n_inpt=n_latent,
            n_hiddens_full=self.gen_n_hiddens_full,
            n_interim_channel=1,
            n_hiddens_conv=self.gen_n_hiddens_conv,
            filter_shapes=self.gen_filter_shapes,
            unpool_factors=self.gen_unpool_factors,
            hidden_transfers_full=self.gen_transfers_full,
            hidden_transfers_conv=self.gen_transfers_conv,
            output_height=self.image_height,
            output_width=self.image_width,
            n_output_channel=self.n_channel,
            out_transfer_conv=assumptions.statify_visible,
            out_transfer_full='identity',
            declare=declare)

        # TODO n_inpt is not a reasonable input for a convnet; this is why None
        # is used here instead.
        GenericVariationalAutoEncoder.__init__(
            self, None, n_latent,
            assumptions, rec_class, gen_class, use_imp_weight=use_imp_weight,
            batch_size=batch_size, optimizer=optimizer,
            max_iter=verbose, verbose=verbose)

    def _make_start_exprs(self):
        inpt = T.tensor4('inpt')
        inpt.tag.test_value, = theano_floatx(np.ones(
            (3, self.n_channel, self.image_height, self.image_width)))

        if self.use_imp_weight:
            imp_weight = T.tensor4('imp_weight')
            imp_weight.tag.test_value, = theano_floatx(np.ones(
                (3, self.n_channel, self.image_height, self.image_width)))
        else:
            imp_weight = None

        return inpt, imp_weight


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

            gen_out_sub = theano.clone(self.vae.gen.rnn.output, {self.vae.gen.inpt: gen_inpt_sub})
            self._f_gen_output = self.function([inpt_m1, latent_prior_sample], gen_out_sub, mode='FAST_COMPILE',
                                                on_unused_input ='warn')
            out = self.vae.gen.sample() if not visible_map else self.vae.gen.maximum
            self._f_visible_sample_by_gen_output = self.function([self.vae.gen.rnn.output], out,
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
            p = self.f_gen(S[:prefix_length + i],
                        latent_samples[:prefix_length + i]
                        )[-1, :, :self.n_inpt]
            S[prefix_length + i] = p

        return S[prefix_length + 1:]
