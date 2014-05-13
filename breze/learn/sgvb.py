# -*- coding: utf-8 -*-

"""Module for learning models with stochastic gradient variational Bayes (SGVB).

The method has been introduced in parallel by [SGVB]_ and [DLGM]_.

The general idea is to optimize a variational upper bound on the negative
log-likelihood of the data with stochastic gradient descent.

References
----------

.. [SGVB] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
.. [DLGM] Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic Back-propagation and Variational Inference in Deep Latent Gaussian Models." arXiv preprint arXiv:1401.4082 (2014).
"""

import theano
import theano.tensor as T
import numpy as np

from breze.arch.component.common import supervised_loss
from breze.arch.component.distributions.mvn import logpdf
from breze.arch.component.misc import inter_gauss_kl
from breze.arch.component.transfer import sigmoid, diag_gauss
from breze.arch.component.varprop.loss import diag_gaussian_nll as diag_gauss_nll
from breze.arch.model import sgvb
from breze.arch.model.neural import mlp
from breze.arch.model.varprop import rnn as vprnn
from breze.arch.model.rnn import rnn
from breze.arch.util import ParameterSet, Model
from breze.learn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)
from breze.learn.utils import theano_floatx


def normal_logpdf(xs, means, vrs):
    residual = xs - means
    divisor = 2 * vrs
    logz = -np.sqrt(vrs * 2 * np.pi)
    return -(residual ** 2 / divisor) + logz


class VariationalAutoEncoder(Model, UnsupervisedBrezeWrapperBase,
                             TransformBrezeWrapperMixin,
                             ReconstructBrezeWrapperMixin):
    """Class representing a variational auto encoder.

    Described in [SGVB]_.

    Attributes
    ----------

    See parameters of ``__init__``.
    """

    transform_expr_name = 'latent'

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 visible,
                 latent_prior='white_gauss',
                 latent_posterior='diag_gauss',
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

        visible : {'bern', 'diag_gauss'}
            String identifiying the distribution of the visibles given the
            latents.

        latent_posterior : {'diag_gauss'}
            Distribution of the latent given the visibles.

        latent_prior : {'white_gauss'}
            Prior distribution of the latents.

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
        self.n_hiddens_recog = n_hiddens_recog
        self.n_latent = n_latent
        self.n_hiddens_gen = n_hiddens_gen
        self.recog_transfers = recog_transfers
        self.gen_transfers = gen_transfers

        self.visible = visible
        self.latent_prior = latent_prior
        self.latent_posterior = latent_posterior

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        super(VariationalAutoEncoder, self).__init__()

        self.f_latent_mean = None
        self.f_latent_mean_var = None
        self.f_out_from_sample = None
        self.f_rec_loss_of_sample = None
        self.f_mvn_logpdf = None

    def _layer_size_by_dist(self, n_vars, dist):
        """Return the cardinality of the sufficient statistics given we want to
        model ``n_units`` variables of distribution ``dist``.

        For example, a diagonal Gaussian needs two sufficient statistics for a
        single random variable, the mean and the variance. A Bernoulli only
        needs one, which is the probability of it being 0 or 1.

        Parameters
        ----------

        n_vars : int
            Number of variables.

        dist : {'diag_gauss', 'bern'}
            Distribution of the variables.

        Returns
        -------

        n : int
            Number of components of the sufficient statistics.
        """
        if dist == 'diag_gauss':
            return 2 * n_vars
        elif dist == 'bern':
            return n_vars
        raise ValueError('unknown distribution in this case: %s' % dist)

    def _init_pars(self):
        spec = {
            'recog': self._recog_par_spec(),
            'gen': self._gen_par_spec(),
        }
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _recog_par_spec(self):
        """Return the specification of the recognition model."""
        n_code_units = self._layer_size_by_dist(
            self.n_latent, self.latent_posterior)
        return mlp.parameters(self.n_inpt, self.n_hiddens_recog,
                              n_code_units)

    def _recog_exprs(self, inpt):
        """Return the exprssions of the recognition model."""
        P = self.parameters.recog

        n_layers = len(self.n_hiddens_recog)
        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(n_layers)]

        if self.latent_posterior == 'diag_gauss':
            out_transfer = diag_gauss
        else:
            raise ValueError('unknown latent posterior distribution:%s'
                             % self.latent_posterior)

        exprs = mlp.exprs(
            inpt, P.in_to_hidden, hidden_to_hiddens, P.hidden_to_out,
            hidden_biases, P.out_bias,
            self.recog_transfers, out_transfer)

        return exprs

    def _gen_par_spec(self):
        """Return the parameter specification of the generating model."""
        n_output = self._layer_size_by_dist(self.n_inpt, self.visible)
        return mlp.parameters(self.n_latent, self.n_hiddens_recog,
                              n_output)

    def _gen_exprs(self, inpt):
        """Return the expression of the generating model."""
        P = self.parameters.gen

        n_layers = len(self.n_hiddens_recog)
        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(n_layers)]

        if self.visible == 'diag_gauss':
            out_transfer = diag_gauss
        elif self.visible == 'bern':
            out_transfer = sigmoid
        else:
            raise ValueError('unknown visible distribution: %s'
                             % self.latent_posterior)
        exprs = mlp.exprs(
            inpt, P.in_to_hidden, hidden_to_hiddens, P.hidden_to_out,
            hidden_biases, P.out_bias,
            self.recog_transfers, out_transfer)

        return exprs

    def _make_inpt(self):
        i = T.matrix('inpt')
        i.tag.test_value, = theano_floatx(np.ones((3, self.n_inpt)))
        return i

    def _make_kl_loss(self):
        E = self.exprs
        n_dim = E['inpt'].ndim

        if self.latent_posterior == 'diag_gauss':
            output = E['recog']['output']
            n_output = output.shape[-1]
            # TODO there is probably a nicer way to do this.
            if n_dim == 3:
                E['latent_mean'] = output[:, :, :n_output // 2]
                E['latent_var'] = output[:, :, n_output // 2:]
            else:
                E['latent_mean'] = output[:, :n_output // 2]
                E['latent_var'] = output[:, n_output // 2:]
        else:
            raise ValueError('unknown latent posterior distribution:%s'
                             % self.latent_posterior)

        if self.latent_posterior == 'diag_gauss' and self.latent_prior == 'white_gauss':
            kl_coord_wise = -inter_gauss_kl(E['latent_mean'], E['latent_var'])
            kl_sample_wise = kl_coord_wise.sum(axis=n_dim - 1)

            kl = kl_sample_wise.mean()
        else:
            raise ValueError(
                'unknown combination for latent_prior and latent_posterior:'
                ' %s, %s' % (self.latent_prior, self.latent_posterior))

        return {
            'kl': kl,
            'kl_coord_wise': kl_coord_wise,
            'kl_sample_wise': kl_sample_wise,
        }

    def _init_exprs(self):
        E = self.exprs = {'inpt': self._make_inpt()}
        n_dim = E['inpt'].ndim

        # Make the expression of the model.
        E.update(sgvb.exprs(
            E['inpt'],
            self._recog_exprs, self._gen_exprs,
            self.visible, self.latent_posterior))

        # Create the reconstruction part of the loss.
        if self.visible == 'diag_gauss':
            loss = diag_gauss_nll
        elif self.visible == 'bern':
            loss = 'bern_ces'
        else:
            raise ValueError('unknown distribution for visibles: %s'
                             % self.visible)

        # TODO this is not going to work with variance propagation.
        rec_loss = supervised_loss(
            E['inpt'], E['gen']['output'], loss, prefix='rec_',
            coord_axis=n_dim - 1)

        # Create the KL divergence part of the loss.
        kl_loss = self._make_kl_loss()

        E.update(rec_loss)
        E.update(kl_loss)

        E.update({
            'loss_sample_wise': E['kl_sample_wise'] + E['rec_loss_sample_wise'],
            'loss': E['kl'] + E['rec_loss'],
        })

    def _latent_mean(self, X):
        # TODO this is only necessary for a Gaussian assumption.
        if self.f_latent_mean is None:
            self.f_latent_mean = self.function(['inpt'], 'latent_mean')
        return self.f_latent_mean(X)

    def _output_from_sample(self, S):
        if self.f_out_from_sample is None:
            self.f_out_from_sample = self.function(['sample'], 'output')
        return self.f_out_from_sample(S)

    def _rec_loss_of_sample(self, X, S):
        if self.f_rec_loss_of_sample is None:
            self.f_rec_loss_of_sample = self.function(
                ['inpt', 'sample'], 'rec_loss_sample_wise')
        return self.f_rec_loss_of_sample(X, S)

    def _latent_mean_var(self, X):
        if self.f_latent_mean_var is None:
            self.f_latent_mean_var = self.function(
                ['inpt'], ['latent_mean', 'latent_var'])
        return self.f_latent_mean_var(X)

    def _mvn_logpdf(self, sample, mean, cov):
        if self.f_mvn_logpdf is None:
            s = T.matrix()
            m = T.vector()
            c = T.matrix()
            mvnlogpdf = logpdf(s, m, c)
            self.f_mvnlogpdf = theano.function([s, m, c], mvnlogpdf)
        return self.f_mvnlogpdf(sample, mean, cov)

    def denoise(self, X):
        """Denoise data from the input distribution.

        The denoising is done as follows. The recognition model is used to
        obtain the posterior of the latent variables. The mode of that is then
        fed through the generating model, of which the result is returned.

        Parameters
        ----------

        X : array_like
            Input to denoise.

        Returns
        -------

        Y : array_like
            Denoised input. Same shape as ``X``.
        """
        H = self._latent_mean(X)
        return self._output_from_sample(H)

    def _estimate_one_nll(self, x, n_samples):
        mean, var = self._latent_mean_var(x[np.newaxis])

        s = np.random.standard_normal((n_samples, mean.shape[1]))
        s *= np.sqrt(var)
        s += mean
        s, = theano_floatx(s)

        targets = np.concatenate([x[np.newaxis]] * n_samples)

        # TODO this is an implcity Gaussian assumption
        lp_v_given_s = -self._rec_loss_of_sample(targets, s)
        lq_s_given_v = self._mvn_logpdf(s, mean[0], np.diag(var.flatten()))
        lp_s = self._mvn_logpdf(
            s, np.zeros(s.shape[1]).astype('float32'),
            np.eye(s.shape[1]).astype('float32'))

        ll = (lp_v_given_s + lp_s - lq_s_given_v).mean()

        return -ll

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
        nll = np.empty(X.shape[0])
        for i, x in enumerate(X):
            nll[i] = self._estimate_one_nll(x, n_samples)
        return nll


class VariationalSequenceAE(VariationalAutoEncoder):

    def _recog_par_spec(self):
        """Return the specification of the recognition model."""
        n_code_units = self._layer_size_by_dist(
            self.n_latent, self.latent_posterior)
        spec = rnn.parameters(self.n_inpt, self.n_hiddens_recog,
                              n_code_units)
        return spec

    def _recog_exprs(self, inpt):
        """Return the exprssions of the recognition model."""
        P = self.parameters.recog

        n_layers = len(self.n_hiddens_recog)
        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(n_layers)]
        initial_hiddens = [getattr(P, 'initial_hiddens_%i' % i)
                           for i in range(n_layers)]
        recurrents = [getattr(P, 'recurrent_%i' % i)
                      for i in range(n_layers)]

        if self.latent_posterior == 'diag_gauss':
            out_transfer = diag_gauss
        else:
            raise ValueError('unknown latent posterior distribution:%s'
                             % self.latent_posterior)

        exprs = rnn.exprs(
            inpt, P.in_to_hidden, hidden_to_hiddens, P.hidden_to_out,
            hidden_biases, initial_hiddens, recurrents, P.out_bias,
            self.recog_transfers, out_transfer)

        return exprs

    def _gen_par_spec(self):
        """Return the parameter specification of the generating model."""
        n_output = self._layer_size_by_dist(self.n_inpt, self.visible)
        return rnn.parameters(self.n_latent, self.n_hiddens_recog,
                              n_output)

    def _gen_exprs(self, inpt):
        """Return the expression of the generating model."""
        P = self.parameters.gen

        n_layers = len(self.n_hiddens_gen)
        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        recurrents = [getattr(P, 'recurrent_%i' % i)
                      for i in range(n_layers)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(n_layers)]
        initial_hiddens = [getattr(P, 'initial_hiddens_%i' % i)
                           for i in range(n_layers)]

        if self.visible == 'diag_gauss':
            out_transfer = diag_gauss
        elif self.visible == 'bern':
            out_transfer = sigmoid
        else:
            raise ValueError('unknown visible distribution: %s'
                             % self.latent_posterior)
        exprs = rnn.exprs(
            inpt, P.in_to_hidden, hidden_to_hiddens, P.hidden_to_out,
            hidden_biases, initial_hiddens, recurrents, P.out_bias,
            self.gen_transfers, out_transfer)

        return exprs

    def _make_inpt(self):
        i = T.tensor3('inpt')
        i.tag.test_value, = theano_floatx(np.ones((3, 2, self.n_inpt)))
        return i

    def _make_kl_loss(self):
        E = self.exprs
        n_dim = E['inpt'].ndim

        if self.latent_posterior == 'diag_gauss':
            output = E['recog']['output']
            n_output = output.shape[-1]
            # TODO there is probably a nicer way to do this.
            if n_dim == 3:
                E['latent_mean'] = output[:, :, :n_output // 2]
                E['latent_var'] = output[:, :, n_output // 2:]
            else:
                E['latent_mean'] = output[:, :n_output // 2]
                E['latent_var'] = output[:, n_output // 2:]
        else:
            raise ValueError('unknown latent posterior distribution:%s'
                             % self.latent_posterior)

        if self.latent_posterior == 'diag_gauss' and self.latent_prior == 'slow_white_gauss':
            d_latent_mean = E['latent_mean'][1:] - E['latent_mean'][:-1]
            d_latent_var = E['latent_var'][1:] + E['latent_var'][:-1]

            kl_first = inter_gauss_kl(E['latent_mean'][:1], E['latent_var'][:1])
            kl_diff = inter_gauss_kl(d_latent_mean, d_latent_var)
            kl_coord_wise = T.concatenate([kl_first, kl_diff])
            kl_sample_wise = kl_coord_wise.sum(axis=2)
            kl = kl_sample_wise.mean()
        elif self.latent_posterior == 'diag_gauss' and self.latent_prior == 'white_gauss':
            kl_coord_wise = inter_gauss_kl(E['latent_mean'], E['latent_var'])
            kl_sample_wise = kl_coord_wise.sum(axis=n_dim - 1)

            kl = kl_sample_wise.mean()
        else:
            raise ValueError(
                'unknown combination for latent_prior and latent_posterior:'
                ' %s, %s' % (self.latent_prior, self.latent_posterior))

        return {
            'kl': kl,
            'kl_coord_wise': kl_coord_wise,
            'kl_sample_wise': kl_sample_wise,
        }

    def _lp_h(self, latent):
        """Return the prior log probability of latent sequences."""
        h_0 = latent[0]
        d_h1_ht = latent[1:] - latent[:-1]

        lp_h_0 = normal_logpdf(
            h_0, np.zeros_like(h_0), np.ones_like(h_0)).sum(axis=1)
        lp_d_h_1_t = normal_logpdf(
            d_h1_ht, np.zeros_like(d_h1_ht), np.ones_like(d_h1_ht)).sum(axis=2)

        return np.concatenate([lp_h_0[np.newaxis], lp_d_h_1_t])

    def _lq_h_given_v(self, latent, visible):
        """Return the log propability of latent variables under the recognition
        model given the visibles."""
        mean, var = self._latent_mean_var(visible)
        lp = normal_logpdf(latent, mean, var)
        return lp.sum(axis=2)

    def _lp_v_given_s(self, visible, latent):
        """Return the log probability of the visibles given the latent
        variables."""
        return -self._rec_loss_of_sample(visible, latent)

    def _estimate_one_nll(self, x, n_samples=10):
        m, v = self._latent_mean_var(x)
        lik = 0
        for i in range(n_samples):
            s = np.random.standard_normal(m.shape) * np.sqrt(v) + m
            s, = theano_floatx(s)
            lp_v_giv_s = self._lp_v_given_s(x, s)
            lp_s = self._lp_h(s)
            lq_s_giv_v = self._lq_h_given_v(s, x)

            loglik = lp_v_giv_s + lp_s - lq_s_giv_v
            lik += np.exp(loglik)
        return -np.log(lik / n_samples)

    def estimate_nll(self, xs, n_samples=10):
        nlls = []
        for x in xs:
            nlls.append(self._estimate_one_nll(self, x[:, np.newaxis], n_samples))
        return nlls


class VariationalFDSequenceAE(VariationalSequenceAE):

    def _recog_par_spec(self):
        """Return the specification of the recognition model."""
        spec = rnn.parameters(self.n_inpt, self.n_hiddens_recog,
                              self.n_latent)
        return spec

    def _recog_exprs(self, inpt):
        """Return the exprssions of the recognition model."""
        P = self.parameters.recog

        n_layers = len(self.n_hiddens_recog)
        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(n_layers)]
        initial_hiddens = [getattr(P, 'initial_hiddens_%i' % i)
                           for i in range(n_layers)]
        recurrents = [getattr(P, 'recurrent_%i' % i)
                      for i in range(n_layers)]

        if self.latent_posterior == 'diag_gauss':
            out_transfer = 'identity'
        else:
            raise ValueError('unknown latent posterior distribution:%s'
                             % self.latent_posterior)

        exprs = vprnn.exprs(
            inpt, T.zeros_like(inpt), P.in_to_hidden, hidden_to_hiddens, P.hidden_to_out,
            hidden_biases, [T.ones_like(b) for b in hidden_biases], initial_hiddens, recurrents,
            P.out_bias, T.ones_like(P.out_bias), self.recog_transfers, out_transfer,
            p_dropouts=[0.1] + len(self.n_hiddens_recog) * [.2])

        return exprs
