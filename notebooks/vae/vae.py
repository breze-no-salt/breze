# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

from breze.arch.model.neural import mlp
from breze.learn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)
from breze.arch.util import ParameterSet, Model
from breze.arch.component.common import supervised_loss
from breze.arch.component.varprop.loss import diag_gaussian_nll as diag_gauss_nll
from breze.arch.component.transfer import sigmoid


def diag_gauss(inpt):
    half = inpt.shape[1] / 2
    mean, var = inpt[:, :half], inpt[:, half:]
    return T.concatenate([mean, var ** 2], axis=1)


def inter_gauss_kl(mean, var):
    return 1 + T.log(var + 1e-4) - mean ** 2 - var


def exprs(inpt, recog_exprs_func, gen_exprs_func, visible_dist,
          latent_posterior_dist,
          latent_key='output', visible_key='output'):
    recog_exprs = recog_exprs_func(inpt)

    latent = recog_exprs[latent_key]

    if latent_posterior_dist == 'diag_gauss':
        n_latent = recog_exprs[latent_key].shape[1] // 2
        latent_mean = latent[:, :n_latent]
        latent_var = latent[:, n_latent:]
        rng = T.shared_randomstreams.RandomStreams()
        noise = rng.normal(size=latent_mean.shape)
        sample = latent_mean + T.sqrt(latent_var + 1e-8) * noise
    else:
        raise ValueError('unknown latent posterior distribution %s' %
                         latent_posterior_dist)

    gen_exprs = gen_exprs_func(sample)
    output = gen_exprs[visible_key]

    return {
        'recog': recog_exprs,
        'gen': gen_exprs,
        'sample': sample,
        'output': output,
    }


class VariationalAutoEncoder(Model, UnsupervisedBrezeWrapperBase,
                             TransformBrezeWrapperMixin,
                             ReconstructBrezeWrapperMixin):

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 visible,
                 latent_prior='white_gauss',
                 latent_posterior='diag_gauss',
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):
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

    def _layer_size_by_dist(self, n_units, dist):
        if dist == 'diag_gauss':
            return 2 * n_units
        elif dist == 'bern':
            return n_units
        raise ValueError('unknown distribution in this case: %s'
                         % dist)

    def _init_pars(self):
        spec = {
            'recog': self._recog_par_spec(),
            'gen': self._gen_par_spec(),
        }
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _recog_par_spec(self):
        n_code_units = self._layer_size_by_dist(
            self.n_latent, self.latent_posterior)
        return mlp.parameters(self.n_inpt, self.n_hiddens_recog,
                              n_code_units)

    def _recog_exprs(self, inpt):
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
        n_output = self._layer_size_by_dist(self.n_inpt, self.visible)
        return mlp.parameters(self.n_latent, self.n_hiddens_recog,
                              n_output)

    def _gen_exprs(self, inpt):
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

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
        }
        self.exprs.update(exprs(
            self.exprs['inpt'],
            self._recog_exprs, self._gen_exprs,
            self.visible, self.latent_posterior))

        if self.visible == 'diag_gauss':
            loss = diag_gauss_nll
        elif self.visible == 'bern':
            loss = 'bern_ces'
        else:
            raise ValueError('unknown distribution for visibles: %s'
                             % self.visible)

        rec_loss = supervised_loss(
            self.exprs['inpt'], self.exprs['gen']['output'], loss,
            prefix='rec_')

        E = self.exprs

        if self.latent_posterior == 'diag_gauss':
            output = E['recog']['output']
            n_output = output.shape[1]
            E['latent_mean'] = output[:, :n_output / 2]
            E['latent_var'] = output[:, n_output / 2:]
        else:
            raise ValueError('unknown latent posterior distribution:%s'
                             % self.latent_posterior)

        if self.latent_posterior == 'diag_gauss' and self.latent_prior == 'white_gauss':
            kl_coord_wise = -inter_gauss_kl(E['latent_mean'], E['latent_var'])
            kl_coord_wise /= 2.
            kl_sample_wise = kl_coord_wise.sum(axis=1)
            kl = kl_sample_wise.mean()
        else:
            raise ValueError(
                'unknown combination for latent_prior and latent_posterior:'
                ' %s, %s' % (self.latent_prior, self.latent_posterior))

        E.update(rec_loss)
        E.update({
            'kl_coord_wise': kl_coord_wise,
            'kl_sample_wise': kl_sample_wise,
            'kl': kl})

        E.update({
            'loss_coord_wise': E['kl_coord_wise'] + E['rec_loss_coord_wise'],
            'loss_sample_wise': E['kl_sample_wise'] + E['rec_loss_sample_wise'],
            'loss': E['kl'] + E['rec_loss'],
        })
