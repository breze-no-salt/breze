# -*- coding: utf-8 -*-

import theano.tensor as T
import numpy as np

from breze.arch.construct.base import Layer
from breze.arch.construct.neural import Mlp, FastDropoutMlp
from breze.arch.component.transfer import diag_gauss
from breze.arch.component.varprop.loss import (
    diag_gaussian_nll as diag_gauss_nll, bern_ces)
from breze.arch.component.common import supervised_loss

from breze.arch.util import get_named_variables

def assert_no_time(X):
    if X.ndim == 2:
        return X
    if X.ndim != 3:
        raise ValueError('ndim must be 2 or 3, but it is %i' % X.ndim)
    return wild_reshape(X, (-1, X.shape[2]))

def normal_logpdf(xs, means, vrs):
    energy = -(xs - means) ** 2 / (2 * vrs)
    partition_func = -T.log(T.sqrt(2 * np.pi * vrs))
    return partition_func + energy

class Distribution(Layer):

    def __init__(self, declare=None, name=None, rng=None):
        if rng is None:
            self.rng = T.shared_randomstreams.RandomStreams()
        else:
            self.rng = rng

        super(Distribution, self).__init__(declare, name)

    def sample(self):
        raise NotImplemented()

    def nll(self, X, inpt=None):
        raise NotImplemented()

class DiagGauss(Distribution):

    def __init__(self, inpt, declare=None, name=None, rng=None):
        self.inpt = inpt
        super(DiagGauss, self).__init__(declare, name, rng)

    def sample(self):
        stt = self.output
        stt_flat = assert_no_time(stt)
        n_latent = stt_flat.shape[1] // 2
        latent_mean = stt_flat[:, :n_latent]
        latent_var = stt_flat[:, n_latent:]
        noise = self.rng.normal(size=latent_mean.shape)
        sample = latent_mean + T.sqrt(latent_var) * noise
        if stt.ndim == 3:
            return recover_time(sample, stt.shape[0])
        else:
            return sample

    def nll(self, X, inpt=None):
        n_dim = self.inpt.ndim
        return supervised_loss(
            X, self.output, lambda x,o: diag_gauss_nll(x,o,1e-4),
            coord_axis=n_dim - 1)['loss_coord_wise']

class NormalGauss(Distribution):

    def __init__(self, inpt, declare=None, name=None, rng=None):
        self.inpt = inpt
        super(NormalGauss, self).__init__(declare, name, rng)

    def _forward(self):
        pass

    def sample(self):
        return self.rng.normal(size=self.inpt.shape)

    def nll(self, X, inpt=None):
        X_flat = X.flatten()
        nll = -normal_logpdf(X_flat, T.zeros_like(X_flat), T.ones_like(X_flat))
        return nll.reshape(X.shape)

class Bernoulli(Distribution):

    def __init__(self, inpt, declare=None, name=None, rng=None):
        self.inpt = inpt
        super(Bernoulli, self).__init__(declare, name, rng)

    def sample(self):
        noise = rng.uniform(size=self.output.shape)
        sample = noise < self.output
        return sample

    def nll(self, X, inpt=None):
        n_dim = 2
        return supervised_loss(
            X, self.output, bern_ces,
            coord_axis=n_dim - 1)['loss_coord_wise']


class MlpDiagGauss(DiagGauss):

    def __init__(self, inpt, n_inpt, n_hiddens, n_output,
                 hidden_transfers, declare=None, name=None, rng=None):

        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers

        super(MlpDiagGauss, self).__init__(inpt, declare, name, rng)

    def _forward(self):
        self.mlp = Mlp(self.inpt, self.n_inpt, self.n_hiddens, self.n_output*2,
                 self.hidden_transfers, diag_gauss, declare=self.declare)

        self.layers = self.mlp.layers
        self.output = self.mlp.output

class FastDropoutMlpDiagGauss(DiagGauss):

    def __init__(self, inpt, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, p_dropout_inpt,
                 p_dropout_hiddens, dropout_parameterized=False, declare=None, name=None, rng=None):
        if rng is None:
            self.rng = T.shared_randomstreams.RandomStreams()
        else:
            self.rng = rng

        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        self.dropout_parameterized = dropout_parameterized

        super(FastDropoutMlpDiagGauss, self).__init__(declare, name)

    def _forward(self):
        self.mlp = FastDropoutMlp(self.inpt, self.n_inpt, self.n_hiddens,
                    self.n_output, self.hidden_transfers,
                    self.out_transfer, self.p_dropout_inpt,
                    self.p_dropout_hiddens, dropout_parameterized=self.dropout_parameterized, declare=self.declare)

        self.layers = self.mlp.layers
        self.output = self.mlp.output

class MlpBernoulli(Bernoulli):

    def __init__(self, inpt, n_inpt, n_hiddens, n_output,
                 hidden_transfers, declare=None, name=None, rng=None):
        if rng is None:
            self.rng = T.shared_randomstreams.RandomStreams()
        else:
            self.rng = rng

        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers

        super(MlpBernoulli, self).__init__(declare, name)

    def _forward(self):
        self.mlp = Mlp(self.inpt, self.n_inpt, self.n_hiddens, self.n_output,
                 self.hidden_transfers, 'sigmoid', declare=self.declare)

        self.layers = self.mlp.layers
        self.output = self.mlp.output
