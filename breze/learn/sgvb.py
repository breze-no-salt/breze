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

# TODO assumption classes
# TODO rename denoise to map denoise
# TODO fix estimation of nll
# TODO rename recurrent models to the ones used in the paper
# TODO make function for missing value imputation

import theano
import theano.tensor as T
import theano.tensor.nnet
import numpy as np

from theano.compile import optdb

from scipy.misc import logsumexp

from breze.arch.component.common import supervised_loss
from breze.arch.component.misc import inter_gauss_kl
from breze.arch.component.transfer import diag_gauss
from breze.arch.component.varprop.transfer import sigmoid
from breze.arch.component.varprop.loss import (
    diag_gaussian_nll as diag_gauss_nll, bern_ces)

from breze.arch.util import ParameterSet
from breze.learn.utils import theano_floatx

import climin.initialize
from climin import mathadapt as ma


from breze.arch.construct.sgvb import VariationalAutoEncoder as _VariationalAutoEncoder

from breze.learn.base import (
    UnsupervisedModel, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)

from breze.arch.construct import neural


# TODO find a better home for the following functions.


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


def wild_reshape(tensor, shape):
    n_m1 = shape.count(-1)
    if n_m1 > 1:
        raise ValueError(' only one -1 allowed in shape')
    elif n_m1 == 1:
        rest = tensor.size
        for s in shape:
            if s != -1:
                rest = rest // s
        shape = tuple(i if i != -1 else rest for i in shape)
    return tensor.reshape(shape)


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


class Assumptions(object):

    def sample_latents(self, stt, rng):
        """Given the output statistis of the recognition model, return samples
        from that distribution."""
        # TODO docstring
        raise NotImplemented()

    def kl_recog_prior(self, stt):
        """Given the output statistics of the recogition model, return the KL
        divergence to the prior."""
        # TODO docstring
        raise NotImplemented()

    def nll_prior(self, X):
        """Given some X, calculate the negative log likelihood of that X under
        the prior."""
        # TODO docstring
        raise NotImplemented()

    def nll_recog_model(self, Z, stt):
        """Given the output statistics of the recognition model, return the
        probability of latent variables Z."""
        # TODO docstring
        raise NotImplemented()

    def nll_gen_model(self, X, stt):
        """Given the output statistics of the generating model, return the
        probability of observed variables X."""
        # TODO docstring
        raise NotImplemented()

    def statify_latent(self, X):
        """Given values X, transform them into valid sufficient statistics for
        the latent distribution."""
        # TODO docstring
        raise NotImplemented()

    def statify_visible(self, X):
        """Given values X, transform them into valid sufficient statistics for
        the visible distribution."""
        # TODO docstring
        raise NotImplemented()


class DiagGaussLatentAssumption(object):

    def statify_latent(self, X, var=None):
        if var is None:
            return diag_gauss(X)
        else:
            return X, var

    def nll_recog_model(self, Z, stt):
        return diag_gauss_nll(Z, stt, 1e-4)

    def kl_recog_prior(self, stt):
        if stt.ndim == 3:
            stt_flat = wild_reshape(stt, (-1, stt.shape[2]))
        else:
            stt_flat = stt

        mean = stt_flat[:, :stt_flat.shape[1] // 2]
        var = stt_flat[:, stt_flat.shape[1] // 2:]
        kl = inter_gauss_kl(mean, var, 1e-4)

        if stt.ndim == 3:
            kl = recover_time(kl, stt.shape[0])

        return kl

    def nll_prior(self, X):
        X_flat = X.flatten()
        nll = -normal_logpdf(X_flat, T.zeros_like(X_flat), T.ones_like(X_flat))
        return nll.reshape(X.shape)

    def latent_layer_size(self, n_latents):
        """Return the cardinality of the sufficient statistics given we want to
        model ``n_latents`` variables.

        For example, a diagonal Gaussian needs two sufficient statistics for a
        single random variable, the mean and the variance. A Bernoulli only
        needs one, which is the probability of it being 0 or 1."""
        return n_latents * 2

    def sample_latents(self, stt, rng):
        stt_flat = assert_no_time(stt)
        n_latent = stt_flat.shape[1] // 2
        latent_mean = stt_flat[:, :n_latent]
        latent_var = stt_flat[:, n_latent:]
        noise = rng.normal(size=latent_mean.shape)
        sample = latent_mean + T.sqrt(latent_var) * noise
        if stt.ndim == 3:
            return recover_time(sample, stt.shape[0])
        else:
            return sample


class DiagGaussVisibleAssumption(object):

    def statify_visible(self, X, var=None):
        if var is None:
            return diag_gauss(X)
        else:
            return X, var

    def nll_gen_model(self, X, stt):
        return diag_gauss_nll(X, stt, 1e-4)

    def visible_layer_size(self, n_latents):
        """Return the cardinality of the sufficient statistics given we want to
        model ``n_latents`` variables.

        For example, a diagonal Gaussian needs two sufficient statistics for a
        single random variable, the mean and the variance. A Bernoulli only
        needs one, which is the probability of it being 0 or 1."""
        return n_latents

    def sample_visibles(self, stt, rng):
        stt_flat = assert_no_time(stt)
        n_latent = stt_flat.shape[1] // 2
        latent_mean = stt_flat[:, :n_latent]
        latent_var = stt_flat[:, n_latent:]
        noise = rng.normal(size=latent_mean.shape)
        sample = latent_mean + T.sqrt(latent_var) * noise
        if stt.ndim == 3:
            return recover_time(sample, stt.shape[0])
        else:
            return sample

    def mode_visibles(self, stt):
        stt_flat = assert_no_time(stt)
        n_latent = stt_flat.shape[1] // 2
        sample = stt_flat[:, :n_latent]
        if stt.ndim == 3:
            return recover_time(sample, stt.shape[0])
        else:
            return sample


class ConstantVarVisibleGaussAssumption(DiagGaussVisibleAssumption):

    out_std = 1

    def statify_visible(self, X, var=None):
        if var is None:
            raise NotImplemented()
        else:
            return X, T.ones_like(var) * self.out_std


class BernoulliVisibleAssumption(object):

    def statify_visible(self, X, var=None):
        if var is not None:
            return sigmoid(X, var)
        else:
            return sigmoid(X, T.zeros_like(X))

    def nll_gen_model(self, X, stt):
        return bern_ces(X, stt)

    def visible_layer_size(self, n_visibles):
        """Return the cardinality of the sufficient statistics given we want to
        model ``n_visibles`` variables.

        For example, a diagonal Gaussian needs two sufficient statistics for a
        single random variable, the mean and the variance. A Bernoulli only
        needs one, which is the probability of it being 0 or 1."""
        return n_visibles

    def sample_visibles(self, stt, rng):
        stt_flat = assert_no_time(stt)
        n_latent = stt_flat.shape[1] // 2
        stt_flat = stt_flat[:, :n_latent]
        noise = rng.uniform(size=stt_flat.shape)
        sample = noise < stt_flat
        if stt.ndim == 3:
            return recover_time(sample, stt.shape[0])
        else:
            return sample

    def mode_visibles(self, stt):
        stt_flat = assert_no_time(stt)
        n_latent = stt_flat.shape[1] // 2
        stt_flat = stt_flat[:, :n_latent]
        sample = stt_flat > 0.5
        if stt.ndim == 3:
            return recover_time(sample, stt.shape[0])
        else:
            return sample


class GenericVariationalAutoEncoder(
    UnsupervisedModel, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin):
    """Class representing a variational auto encoder.

    Described in [SGVB]_.

    Attributes
    ----------

    See parameters of ``__init__``.
    """

    shortcut = None

    def __init__(self, n_inpt, n_latent,
                 assumptions, gen_class, rec_class, condition_func=None,
                 use_imp_weight=False,
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):
        """Create a VariationalAutoEncoder object.

        Parameters
        ----------

        n_inpt : int
            Dimensionality of the visibles.

        n_hiddens_recog : list of integers
            List containing the sizes of the hidden layers of the recognition
            model.

        n_latent : int
            Dimensionality of the latent dimension.

        n_hiddens_gen : list of integers
            List containing the sizes of the hidden layers of the generating
            model.

        recog_transfers : list
            List containing the transfer functions for the hidden layers of the
            recognition model.

        gen_transfers : list
            List containing the transfer cuntions for the hidden layers of the
            generating model.

        assumptions : Assumptions object
            Object encoding the assumptions about the data.

        imp_weight : boolean
            Flag indicating whether importance weights are used.

        batch_size : int
            Size of each mini batch during training.

        optimizer: string or pair
            Can be either a string or a pair. In any case,
            ``climin.util.optimizer`` is used to construct an optimizer. In the
            case of a string, the string is used as an identifier for the
            optimizer which is then instantiated with default arguments. If a
            pair, expected to be (`identifier`, `kwargs`) for more fine control
            of the optimizer.

        max_iter : integer
            Maximum number of optimization iterations to perform.

        verbose : boolean
            Flag indicating whether to print out information during fitting.
        """
        self.n_inpt = n_inpt
        self.n_latent = n_latent
        self.n_output = assumptions.visible_layer_size(n_inpt)
        self.assumptions = assumptions
        self.rec_class = rec_class
        self.gen_class = gen_class
        self.condition_func = condition_func

        self.use_imp_weight = use_imp_weight
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        self.f_latent_mean = None
        self.f_latent_mean_var = None
        self.f_out_from_sample = None
        self.f_rec_loss_of_sample = None
        self.f_mvn_logpdf = None
        self.f_estimate_nll = None

        self._init_exprs()

    def _make_start_exprs(self):
        inpt = T.matrix('inpt')
        inpt.tag.test_value, = theano_floatx(np.ones((3, self.n_inpt)))

        if self.use_imp_weight:
            imp_weight = T.matrix('imp_weight')
            imp_weight.tag.test_value, = theano_floatx(np.ones((3, 1)))
        else:
            imp_weight = None

        return inpt, imp_weight

    def _init_exprs(self):
        inpt, self.imp_weight = self._make_start_exprs()
        parameters = ParameterSet()

        n_dim = inpt.ndim

        self.vae = _VariationalAutoEncoder(inpt, self.n_inpt,
                                           self.n_latent, self.n_output,
                                           self.assumptions,
                                           self.gen_class,
                                           self.rec_class,
                                           self.condition_func,
                                           declare=parameters.declare)

        if self.use_imp_weight:
            imp_weight = T.addbroadcast(self.imp_weight, n_dim - 1)
        else:
            imp_weight = False

        rec_loss = supervised_loss(
            inpt, self.vae.output, self.assumptions.nll_gen_model,
            coord_axis=n_dim - 1,
            imp_weight=imp_weight)['loss_coord_wise']
        self.rec_loss_sample_wise = rec_loss.sum(axis=n_dim - 1)
        self.rec_loss = self.rec_loss_sample_wise.mean()

        output = self.vae.gen.output
        self.latent = self.vae.recog.output
        self.sample = self.vae.sample

        # Create the KL divergence part of the loss.
        n_dim = inpt.ndim
        self.kl_coord_wise = self.assumptions.kl_recog_prior(self.latent)

        if self.use_imp_weight:
            self.kl_coord_wise *= imp_weight
        self.kl_sample_wise = self.kl_coord_wise.sum(axis=n_dim - 1)
        self.kl = self.kl_sample_wise.mean()

        self.loss_sample_wise = self.kl_sample_wise + self.rec_loss_sample_wise
        loss = self.kl + self.rec_loss

        UnsupervisedModel.__init__(self, inpt=inpt,
                                 output=output,
                                 loss=loss,
                                 parameters=parameters,
                                 imp_weight=self.imp_weight)

        self.transform_expr_name = self.latent

    def _fix_imp_weight(self, ndim):
        # For the VAE, the importance weights cannot be coordinate
        # wise, but have to be sample wise. In numpy, we can just use an
        # array where the last dimensionality is 1 and the broadcasting
        # rules will make this work as one would expect. In theano's case,
        # we have to be explicit about whether broadcasting along that
        # dimension is allowed though, since normal tensor3s do not allow
        # it. The following code achieves this.
        return T.addbroadcast(self.imp_weight, ndim - 1)

    def _output_from_sample(self, S):
        if self.f_out_from_sample is None:
            self.f_out_from_sample = self.function([self.sample], self.output)
        return self.f_out_from_sample(S)

    def _rec_loss_of_sample(self, X, S):
        if self.f_rec_loss_of_sample is None:
            self.f_rec_loss_of_sample = self.function(
                ['inpt', self.sample], self.rec_loss_sample_wise)
        return self.f_rec_loss_of_sample(X, S)

    def estimate_nll(self, X, n_samples=10):
        """Return an estimate of the negative log-likelihood of ``X``.

        The estimate is obtained via importance sampling, as described in the
        appendix of [DLGM]_.

        Parameters
        ----------

        X : array_like
            Input to estimate the likelihood of.

        n_samples : int
            Number of samples for importance sampling. The more, the more
            reliable the estimator becomes.

        Returns
        -------

        nll : array_like
            Array of shape ``(n,)`` where each entry corresponds to the nll of
            corresponding sample in ``X``.
        """
        if getattr(self, 'f_estimate_nll', None) is None:
            self.f_estimate_nll = self._make_f_estimate_nll()
        return self.f_estimate_nll(X, n_samples)

    def _make_f_estimate_nll(self):
        ndim = self.sample.ndim
        if ndim == 3:
            latent_sample = T.tensor3('sample')
            latent_sample.tag.test_value = np.zeros(
                (self.exprs['inpt'].tag.test_value.shape[0],
                 self.exprs['inpt'].tag.test_value.shape[1],
                 self.n_latent)).astype(theano.config.floatX)
        elif ndim == 2:
            latent_sample = T.matrix('sample')
            latent_sample.tag.test_value = np.zeros(
                (self.exprs['inpt'].tag.test_value.shape[0],
                 self.n_latent)).astype(theano.config.floatX)
        else:
            raise ValueError('unexpected ndim for samples')

        # Map a sample s to the prior log probability p(z)
        nll_z = self.assumptions.nll_prior(latent_sample).sum(axis=ndim - 1)
        f_nll_z = self.function([latent_sample], nll_z, on_unused_input='ignore')

        # Map a given visible x and a sample z to the generating
        # probability p(x|z).
        nll_x_given_z = self.assumptions.nll_gen_model(
            self.inpt, self.output).sum(axis=ndim - 1)
        f_nll_x_given_z = self.function([self.inpt, latent_sample], nll_x_given_z,
                                        givens={self.sample: latent_sample})

        # Map a given visible x and a sample z to the recognition
        # probability q(z|x).
        nll_z_given_x = self.assumptions.nll_recog_model(
            latent_sample, self.latent).sum(axis=ndim - 1)
        f_nll_z_given_x = self.function(
            [latent_sample, self.inpt], nll_z_given_x)

        # Sample some z from q(z|x).
        f_sample_z_given_x = self.function([self.inpt], self.sample)

        def inner(X, n):
            return estimate_nll(X, f_nll_z, f_nll_x_given_z, f_nll_z_given_x,
                                f_sample_z_given_x, n)

        # So it gets deleted before pickling.
        inner.breze_func = True

        return inner


class VariationalAutoEncoder(GenericVariationalAutoEncoder):

    """
        Variational Autoencoder with Mlp as recognition and generative model
    """

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 assumptions,
                 p_dropout_inpt=.1, p_dropout_hiddens=.1,
                 use_imp_weight=False,
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):
        self.n_hiddens_recog = n_hiddens_recog
        self.n_hiddens_gen = n_hiddens_gen
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        self.recog_transfers = recog_transfers
        self.gen_transfers = gen_transfers

        if isinstance(p_dropout_hiddens, float):
            p_dropout_hiddens = [p_dropout_hiddens] * len(n_hiddens_recog)

        rec_class = lambda inpt, declare: neural.FastDropoutMlp(
            inpt, n_inpt,
            n_hiddens_recog,
            n_latent,
            recog_transfers, assumptions.statify_latent,
            p_dropout_inpt, p_dropout_hiddens,
            dropout_parameterized=True,
            declare=declare)

        gen_class = lambda inpt, declare: neural.FastDropoutMlp(
            inpt, n_latent,
            n_hiddens_gen,
            assumptions.visible_layer_size(n_inpt),
            gen_transfers, assumptions.statify_visible,
            p_dropout_inpt, p_dropout_hiddens,
            dropout_parameterized=True,
            declare=declare)

        GenericVariationalAutoEncoder.__init__(self, n_inpt, n_latent,
                 assumptions, rec_class, gen_class, use_imp_weight=use_imp_weight,
                 batch_size=batch_size, optimizer=optimizer,
                 max_iter=verbose, verbose=verbose)


class StochasticRnn(GenericVariationalAutoEncoder):

    sample_dim = 1,
    theano_optimizer = optdb.query(theano.gof.Query(
        include=['fast_run'], exclude=['scan_eqopt1', 'scan_eqopt2']))
    mode = theano.Mode(linker='cvm', optimizer=theano_optimizer)

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 assumptions,
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

        if isinstance(p_dropout_hiddens, float):
            p_dropout_hiddens = [p_dropout_hiddens] * len(n_hiddens_recog)
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        if p_dropout_hidden_to_out is None:
            p_dropout_hidden_to_out = p_dropout_hiddens[-1]
        else:
            p_dropout_hidden_to_out = p_dropout_hidden_to_out
        self.p_dropout_hidden_to_out = p_dropout_hidden_to_out

        self.p_dropout_shortcut = p_dropout_shortcut

        rec_class = lambda inpt, declare: neural.FastDropoutRnn(
            inpt, n_inpt,
            n_hiddens_recog,
            n_latent,
            recog_transfers, assumptions.statify_latent,
            p_dropout_inpt='parameterized',
            p_dropout_hiddens=['parameterized' for _ in n_hiddens_recog],
            p_dropout_hidden_to_out='parameterized',
            declare=declare)

        gen_class = lambda inpt, declare: neural.FastDropoutRnn(
            inpt, n_latent + n_inpt,
            n_hiddens_gen,
            assumptions.visible_layer_size(n_inpt),
            gen_transfers, assumptions.statify_visible,
            p_dropout_inpt, p_dropout_hiddens, p_dropout_hidden_to_out,
            declare=declare)

        condition_func = lambda rec: T.concatenate(
            [T.zeros_like(rec.inpt[:1]), rec.inpt[:-1]], 0)
#
#        # Hic sunt dracones.
#        # If we do not keep this line, Theano will die with a segfault.
#        shortcut_empty = T.set_subtensor(T.zeros_like(shortcut)[:, :, :], shortcut)

        GenericVariationalAutoEncoder.__init__(self, n_inpt, n_latent,
            assumptions, rec_class, gen_class, condition_func,
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

    def initialize(self,
                   par_std=1, par_std_affine=None, par_std_rec=None,
                   par_std_in=None,
                   sparsify_affine=None, sparsify_rec=None,
                   spectral_radius=None):
        climin.initialize.randomize_normal(self.parameters.data, 0, par_std)
        all_layers = self.vae.recog.layers + self.vae.gen.layers
        for i, layer in enumerate(all_layers):
            if hasattr(layer, 'recurrent'):
                p = self.parameters[layer.recurrent.weights]
                if par_std_rec:
                    climin.initialize.randomize_normal(p, 0, par_std_rec)
                if sparsify_rec:
                    climin.initialize.sparsify_columns(p, sparsify_rec)
                if spectral_radius:
                    climin.initialize.bound_spectral_radius(p, spectral_radius)
                self.parameters[layer.recurrent.initial_mean][...] = 0
                self.parameters[layer.recurrent.initial_std][...] = 1
            if hasattr(layer, 'affine'):
                p = self.parameters[layer.affine.weights]
                if par_std_affine:
                    if i == 0 and par_std_in:
                        climin.initialize.randomize_normal(p, 0, par_std_in)
                    else:
                        climin.initialize.randomize_normal(p, 0, par_std_affine)
                if sparsify_affine:
                    climin.initialize.sparsify_columns(p, sparsify_affine)

                self.parameters[layer.affine.bias][...] = 0

    def _make_sample_one_step(self, visible_map=False):
        n_layers = len(self.n_hiddens_gen)

        # make clones for parameters, so they can be used as inputs
        initials = []
        for i in range(n_layers):
            initials += ['initial_hidden_means_%i' % i, 'initial_hidden_vars_%i' % i]

        clones = [T.vector(i) for i in initials]
        inpts = clones + [self.exprs['sample'], self.exprs['inpt']]
        gen_exprs = self.exprs['gen']
        outputs = [[gen_exprs['hidden_mean_%i' % i], gen_exprs['hidden_var_%i' % i]]
                   for i in range(n_layers)]
        outputs = flatten_list(outputs)

        visible_stt = self.exprs['gen']['output']
        if visible_map:
            visible_sample = self.assumptions.mode_visibles(visible_stt)
        else:
            rng = T.shared_randomstreams.RandomStreams()
            visible_sample = self.assumptions.sample_visibles(visible_stt, rng)

        # TODO sample here instead of MAP
        outputs += [visible_sample]

        return self.function(
            inpts, outputs, givens=dict(zip([getattr(self.parameters.gen, i) for i in initials],
                                            clones)),
            on_unused_input='warn')

    def _sample_one_step(self, *args):
        if getattr(self, 'f_sample_one_step', None) is None:
            self.f_sample_one_step = self._make_sample_one_step()

        res = self.f_sample_one_step(*args)
        return res

    def _sample_one_step_vmap(self, *args):
        if getattr(self, 'f_sample_one_step_vmap', None) is None:
            self.f_sample_one_step_vmap = self._make_sample_one_step(
                visible_map=True)

        res = self.f_sample_one_step_vmap(*args)
        return res

    def sample(self, n_time_steps, visible_map=False):
        samples = []
        n_layers = len(self.n_hiddens_gen)
        initial_means = [self.parameters['gen']['initial_hidden_means_%i' % i]
                         for i in range(n_layers)]
        initial_vars = [self.parameters['gen']['initial_hidden_vars_%i' % i]
                        for i in range(n_layers)]

        # TODO: sample from assumption prior instead of from standard normal.
        latent_samples = np.random.standard_normal((1, 1, self.n_latent)
                                                   ).astype(theano.config.floatX)
        inpt = np.zeros((1, 1, self.n_inpt), dtype=theano.config.floatX)
        args = flatten_list(zip(initial_means, initial_vars))
        args += [latent_samples, inpt]

        if visible_map:
            sampler = self._sample_one_step_vmap
        else:
            sampler = self._sample_one_step

        for i in range(n_time_steps):
            args = sampler(*args)
            samples.append(args[-1])
            args = [i[0, 0, :] for i in args[:-1]] + [latent_samples] + args[-1:]

        return np.concatenate(samples, axis=1)[0]


class BidirectStochasticRnn(StochasticRnn):

    def _gen_par_spec(self):
        """Return the parameter specification of the generating model."""
        n_output = self.assumptions.visible_layer_size(self.n_inpt)
        spec = vprnn.parameters(
            self.n_latent + self.n_inpt, self.n_hiddens_gen,
            n_output,
            hidden_transfers=self.gen_transfers,
        )
        return spec

    def _recog_par_spec(self):
        """Return the specification of the recognition model."""
        spec = vpbrnn.parameters(self.n_inpt, self.n_hiddens_recog,
                                 self.n_latent)

        spec['p_dropout'] = {
            'inpt': 1,
            'hiddens': [1 for _ in self.n_hiddens_recog],
            'hidden_to_out': 1,
        }

        return spec

    def _recog_exprs(self, inpt):
        """Return the exprssions of the recognition model."""
        P = self.parameters.recog

        n_layers = len(self.n_hiddens_recog)
        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(n_layers)]
        initial_hidden_means_fwd = [
            getattr(P, 'initial_hidden_means_fwd_%i' % i)
            for i in range(n_layers)]
        initial_hidden_vars_fwd = [
            getattr(P, 'initial_hidden_vars_fwd_%i' % i) ** 2 + 1e-4
            for i in range(n_layers)]
        initial_hidden_means_bwd = [
            getattr(P, 'initial_hidden_means_bwd_%i' % i)
            for i in range(n_layers)]
        initial_hidden_vars_bwd = [
            getattr(P, 'initial_hidden_vars_bwd_%i' % i) ** 2 + 1e-4
            for i in range(n_layers)]
        recurrents_fwd = [getattr(P, 'recurrent_fwd_%i' % i)
                          for i in range(n_layers)]
        recurrents_bwd = [getattr(P, 'recurrent_bwd_%i' % i)
                          for i in range(n_layers)]

        p_dropouts = (
            [P.p_dropout.inpt] + P.p_dropout.hiddens
            + [P.p_dropout.hidden_to_out])

        # Reparametrize to assert the rates lie in (0.025, 1-0.025).
        p_dropouts = [T.nnet.sigmoid(i) * 0.95 + 0.025 for i in p_dropouts]

        exprs = vpbrnn.exprs(
            inpt, T.zeros_like(inpt), P.in_to_hidden, hidden_to_hiddens, P.hidden_to_out,
            hidden_biases, [1 for _ in hidden_biases],
            initial_hidden_means_fwd, initial_hidden_vars_fwd,
            initial_hidden_means_bwd, initial_hidden_vars_bwd,
            recurrents_fwd, recurrents_bwd,
            P.out_bias, 1, self.recog_transfers, self.assumptions.statify_latent,
            p_dropouts=p_dropouts)
        exprs['inpt'] = inpt

        #to_shortcut = self.exprs['inpt']
        to_shortcut = self.exprs['inpt']

        shortcut = T.concatenate([T.zeros_like(to_shortcut[:1]),
                                  to_shortcut[:-1]])

        # Hic sunt dracones.
        # If we do not keep this line, Theano will die with a segfault.
        shortcut_empty = T.set_subtensor(T.zeros_like(shortcut)[:, :, :], shortcut)

        exprs['shortcut'] = shortcut_empty

        return exprs

    def draw_pars(self, par_std, par_std_i2h, sparsify_in, sparsify_rec,
                  spectral_radius):
        n_recog_layers = len(self.recog_transfers)
        n_gen_layers = len(self.gen_transfers)

        climin.initialize.randomize_normal(
            self.parameters.data, 0, par_std)
        climin.initialize.randomize_normal(
            self.parameters['recog']['in_to_hidden'],
            0, par_std_i2h)
        climin.initialize.randomize_normal(
            self.parameters['gen']['in_to_hidden'],
            0, par_std_i2h)

        for i in range(n_recog_layers):
            if sparsify_rec:
                climin.initialize.sparsify_columns(
                    self.parameters['recog']['recurrent_fwd_%i' % i], sparsify_rec)
                climin.initialize.sparsify_columns(
                    self.parameters['recog']['recurrent_bwd_%i' % i], sparsify_rec)
            climin.initialize.bound_spectral_radius(
                self.parameters['recog']['recurrent_fwd_%i' % i], spectral_radius)
            climin.initialize.bound_spectral_radius(
                self.parameters['recog']['recurrent_bwd_%i' % i], spectral_radius)

        for i in range(n_gen_layers):
            if sparsify_rec:
                climin.initialize.sparsify_columns(
                    self.parameters['gen']['recurrent_%i' % i], sparsify_rec)
            climin.initialize.bound_spectral_radius(
                self.parameters['gen']['recurrent_%i' % i], spectral_radius)

        if sparsify_in:
            climin.initialize.sparsify_columns(
                self.parameters['recog']['in_to_hidden'], sparsify_in)

        self.parameters['recog']['p_dropout']['hiddens'][0] = -3
        self.parameters['recog']['p_dropout']['hidden_to_out'][0] = -3
        self.parameters['recog']['p_dropout']['inpt'][0] = -3
