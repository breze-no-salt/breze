# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from climin import mathadapt as ma
from scipy.misc import logsumexp

from breze.arch.construct.layer.kldivergence import kl_div
from breze.arch.construct.sgvb import (
    VariationalAutoEncoder as _VariationalAutoEncoder)
from breze.arch.util import ParameterSet
from breze.learn.base import (
    UnsupervisedModel, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)
from breze.learn.utils import theano_floatx


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


class GenericVariationalAutoEncoder(UnsupervisedModel,
                                    TransformBrezeWrapperMixin,
                                    ReconstructBrezeWrapperMixin):
    """Class representing a variational auto encoder.

    Described in [SGVB]_.

    Attributes
    ----------

    See parameters of ``__init__``.
    """

    shortcut = None

    def __init__(self, n_inpt, n_latent,
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
        self.n_output = n_inpt

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
        self.parameters = ParameterSet()

        n_dim = inpt.ndim

        self.vae = _VariationalAutoEncoder(inpt, self.n_inpt,
                                           self.n_latent, self.n_output,
                                           self.make_recog,
                                           self.make_prior,
                                           self.make_gen,
                                           getattr(self, 'make_cond', None),
                                           declare=self.parameters.declare)

        self.recog_sample = self.vae.recog_sample

        if self.use_imp_weight:
            imp_weight = T.addbroadcast(self.imp_weight, n_dim - 1)
        else:
            imp_weight = False

        rec_loss = self.vae.gen.nll(inpt)
        self.rec_loss_sample_wise = rec_loss.sum(axis=n_dim - 1)
        self.rec_loss = self.rec_loss_sample_wise.mean()

        output = self.vae.gen.stt

        # Create the KL divergence part of the loss.
        n_dim = inpt.ndim
        self.kl_coord_wise = kl_div( self.vae.recog, self.vae.prior)

        if self.use_imp_weight:
            self.kl_coord_wise *= imp_weight
        self.kl_sample_wise = self.kl_coord_wise.sum(axis=n_dim - 1)
        self.kl = self.kl_sample_wise.mean()

        # FIXME: this does not work with convolutional aes
        # self.loss_sample_wise = self.kl_sample_wise + self.rec_loss_sample_wise
        loss = self.kl + self.rec_loss

        UnsupervisedModel.__init__(self, inpt=inpt,
                                   output=output,
                                   loss=loss,
                                   parameters=self.parameters,
                                   imp_weight=self.imp_weight)

        # TODO: this has to become transform_expr or sth like that
        # TODO: convert distribution parameters to latent stt
        #self.transform_expr_name = self.vae.latent
        self.transform_expr_name = None

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
            self.f_out_from_sample = self.function(
                [self.vae.recog_sample], self.output)
        return self.f_out_from_sample(S)

    def _rec_loss_of_sample(self, X, S):
        if self.f_rec_loss_of_sample is None:
            self.f_rec_loss_of_sample = self.function(
                ['inpt', self.vae.recog_sample], self.rec_loss_sample_wise)
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
        ndim = self.vae.recog_sample.ndim
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
        nll_z = self.vae.prior.nll(latent_sample).sum(axis=ndim - 1)

        # Depends on the shape of the input otherwis.
        nll_z = theano.clone(nll_z, {self.recog_sample: latent_sample})

        f_nll_z = self.function([latent_sample], nll_z,
                                on_unused_input='ignore')

        # Map a given visible x and a sample z to the generating
        # probability p(x|z).
        nll_x_given_z = self.vae.gen.nll(
            self.inpt, self.output).sum(axis=ndim - 1)
        f_nll_x_given_z = self.function(
            [self.inpt, latent_sample], nll_x_given_z,
            givens={self.vae.recog_sample: latent_sample})

        # Map a given visible x and a sample z to the recognition
        # probability q(z|x).
        nll_z_given_x = self.vae.recog.nll(latent_sample).sum(axis=ndim - 1)
        f_nll_z_given_x = self.function(
            [latent_sample, self.inpt], nll_z_given_x)

        # Sample some z from q(z|x).
        f_sample_z_given_x = self.function([self.inpt], self.vae.recog_sample)

        def inner(X, n):
            return estimate_nll(X, f_nll_z, f_nll_x_given_z, f_nll_z_given_x,
                                f_sample_z_given_x, n)

        # So it gets deleted before pickling.
        inner.breze_func = True

        return inner
