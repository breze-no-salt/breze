# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

from breze.arch.model.neural import mlp
from breze.learn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)
from breze.arch.util import ParameterSet, Model, get_named_variables
from breze.arch.component.common import supervised_loss
from breze.arch.component.varprop.loss import diag_gaussian_nll


def parameters(n_inpt, n_hiddens_enc, n_hiddens_dec,
               visible, latent_posterior):
    if latent_posterior == 'diag_gaussian':
        n_latent_fields = n_hiddens_enc[-1] * 2
    else:
        raise ValueError('unknown latent posterior distribution '
                         '%s' % latent_posterior)

    if visible == 'diag_gaussian':
        n_output = n_inpt * 2
    elif visible == 'bernoulli':
        n_output = n_inpt
    else:
        raise ValueError('unknown visible distribution %s' % visible)

    enc_pars = mlp.parameters(n_inpt, n_hiddens_enc[:-1], n_latent_fields)
    dec_pars = mlp.parameters(n_hiddens_enc[-1], n_hiddens_dec, n_output)

    prefixed_enc_pars = dict(('enc_%s' % k, v) for k, v in enc_pars.items())
    prefixed_dec_pars = dict(('dec_%s' % k, v) for k, v in dec_pars.items())

    prefixed_enc_pars.update(prefixed_dec_pars)

    return prefixed_enc_pars


def exprs(inpt,
          in_to_enc, hidden_to_hidden_enc, hidden_to_latent,
          hidden_biases_enc, latent_bias,
          hidden_transfers_enc,
          latent_to_dec, hidden_to_hidden_dec, hidden_to_output,
          hidden_biases_dec, out_bias,
          hidden_transfers_dec, visible, latent_posterior,
          prefix=''):

    if latent_posterior == 'diag_gaussian':
        latent_transfer = diag_gaussian
    else:
        raise ValueError('unknown latent posterior distribution %s' % latent_posterior)

    encoder_exprs = mlp.exprs(
        inpt, in_to_enc, hidden_to_hidden_enc, hidden_to_latent,
        hidden_biases_enc, latent_bias,
        hidden_transfers_enc, latent_transfer,
        prefix='enc_')

    if latent_posterior == 'diag_gaussian':
        latent = encoder_exprs['enc_output']
        n_latent = latent.shape[1] / 2
        latent_mean, latent_var = latent[:, :n_latent], latent[:, n_latent:]
        encoder_exprs['latent_mean'] = latent_mean
        encoder_exprs['latent_var'] = latent_var

        rng = T.shared_randomstreams.RandomStreams()
        noise = rng.normal(size=latent_mean.shape)
        #noise = T.cast(noise, theano.config.floatX)

        sample = latent_mean + T.sqrt(latent_var + 1e-8) * noise
    else:
        raise ValueError('unknown latent posterior distribution %s' %
                         latent_posterior)

    if visible == 'bernoulli':
        out_transfer = 'sigmoid'
    elif visible == 'diag_gaussian':
        out_transfer = diag_gaussian
    else:
        raise ValueError('unknown visible distribution %s' % visible)

    decoder_exprs = mlp.exprs(
        sample, latent_to_dec, hidden_to_hidden_dec, hidden_to_output,
        hidden_biases_dec, out_bias,
        hidden_transfers_dec, out_transfer,
        prefix='dec_')

    exprs = get_named_variables(locals())
    exprs.update(encoder_exprs)
    exprs.update(decoder_exprs)
    exprs['sample'] = sample

    return exprs


def diag_gaussian(inpt):
    half = inpt.shape[1] / 2
    mean, var = inpt[:, :half], inpt[:, half:]
    return T.concatenate([mean, var ** 2], axis=1)


def inter_gaussian_kl(mean, var):
    return 1 + T.log(var + 1e-4) - mean ** 2 - var


class VariationalAutoEncoder(Model, UnsupervisedBrezeWrapperBase,
                             TransformBrezeWrapperMixin,
                             ReconstructBrezeWrapperMixin):

    def __init__(self, n_inpt, n_hiddens_enc, n_hiddens_dec,
                 enc_transfers, dec_transfers,
                 visible,
                 latent_prior='white_gaussian',
                 latent_posterior='diag_gaussian',
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):
        self.n_inpt = n_inpt
        self.n_hiddens_enc = n_hiddens_enc
        self.n_hiddens_dec = n_hiddens_dec
        self.enc_transfers = enc_transfers
        self.dec_transfers = dec_transfers

        self.visible = visible
        self.latent_prior = latent_prior
        self.latent_posterior = latent_posterior

        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        super(VariationalAutoEncoder, self).__init__()

    def _init_pars(self):
        spec = parameters(
            self.n_inpt, self.n_hiddens_enc, self.n_hiddens_dec,
            self.visible, self.latent_posterior)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
        }
        P = self.parameters

        n_layers_enc = len(self.n_hiddens_enc) - 1
        hidden_to_hiddens_enc = [getattr(P, 'enc_hidden_to_hidden_%i' % i)
                                 for i in range(n_layers_enc - 1)]
        hidden_biases_enc = [getattr(P, 'enc_hidden_bias_%i' % i)
                             for i in range(n_layers_enc)]

        n_layers_dec = len(self.n_hiddens_dec)
        hidden_biases_dec = [getattr(P, 'dec_hidden_bias_%i' % i)
                             for i in range(n_layers_dec)]
        hidden_to_hiddens_dec = [getattr(P, 'dec_hidden_to_hidden_%i' % i)
                                 for i in range(n_layers_dec - 1)]

        self.exprs.update(exprs(
            self.exprs['inpt'],
            P.enc_in_to_hidden, hidden_to_hiddens_enc, P.enc_hidden_to_out,
            hidden_biases_enc, P.enc_out_bias,
            self.enc_transfers,
            P.dec_in_to_hidden, hidden_to_hiddens_dec, P.dec_hidden_to_out,
            hidden_biases_dec, P.dec_out_bias,
            self.dec_transfers, self.visible, self.latent_posterior))

        if self.visible == 'diag_gaussian':
            loss = diag_gaussian_nll
        elif self.visible == 'bernoulli':
            loss = 'bern_ces'

        rec_loss = supervised_loss(self.exprs['inpt'], self.exprs['dec_output'],
                                   loss, prefix='rec_')

        E = self.exprs

        if self.latent_posterior == 'diag_gaussian' and self.latent_prior == 'white_gaussian':
            kl_coord_wise = -inter_gaussian_kl(E['latent_mean'], E['latent_var'])
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
