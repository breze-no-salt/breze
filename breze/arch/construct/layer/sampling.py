# -*- coding: utf-8 -*-

import theano.tensor as T
import numpy as np

from breze.arch.construct.base import Layer
from breze.arch.construct.neural import Mlp
from breze.arch.component.transfer import diag_gauss
from breze.arch.component.varprop.loss import (
    diag_gaussian_nll as diag_gauss_nll, bern_ces)
from breze.arch.component.common import supervised_loss
from breze.arch.component.misc import inter_gauss_kl

from breze.arch.util import get_named_variables

def normal_logpdf(xs, means, vrs):
    energy = -(xs - means) ** 2 / (2 * vrs)
    partition_func = -T.log(T.sqrt(2 * np.pi * vrs))
    return partition_func + energy

class Gauss(Layer):

    def __init__(self, rng=None, name=None):
        if rng is None:
            self.rng = T.shared_randomstreams.RandomStreams()
        else:
            self.rng = rng
        super(Gauss, self).__init__(name)

    def forward(self, mean, var):
        std = T.sqrt(var)
        noise = self.rng.normal(size=mean.shape)
        sample = mean + noise * std

        self.exprs = get_named_variables(locals())
        self.output = [sample]


class GaussProjection(Layer):

    def forward(self, mean, var):
        var = T.sqrt(var ** 2 + 1e-8)
        self.exprs = get_named_variables(locals())
        self.output = [mean, var]

class NormalGauss(Layer):

    def __init__(self, inpt, n_output,
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

        super(MlpDiagGauss, self).__init__(declare, name)

    def _forward(self):
        self.output = T.concatenate((T.zeros((1,self.n_output)),T.ones((1,self.n_output))),1)

    def sample(self):
        return self.rng.normal(size=mean.shape)


class MlpDiagGauss(Layer):

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

        super(MlpDiagGauss, self).__init__(declare, name)

    def _forward(self):
        self.mlp = Mlp(self.inpt, self.n_inpt, self.n_hiddens, self.n_output*2,
                 self.hidden_transfers, diag_gauss, declare=self.declare)

        self.layers = self.mlp.layers
        self.output = self.mlp.output

    def sample(self):
        mean = self.output[:, :self.n_output]
        var = self.output[:, self.n_output:]

        std = T.sqrt(var)
        noise = self.rng.normal(size=mean.shape)
        sample = mean + noise * std
        return sample

    def nll(self, X, inpt=None):
        n_dim = 2
        return supervised_loss(
            X, self.output, lambda x,o: diag_gauss_nll(x,o,1e-4),
            coord_axis=n_dim - 1)['loss_coord_wise']

    def kl_prior(self):
        mean = self.output[:, :self.n_output]
        var = self.output[:, self.n_output:]
        kl = inter_gauss_kl(mean, var, 1e-4)

        return kl

    def nll_prior(self, X):
        X_flat = X.flatten()
        nll = -normal_logpdf(X_flat, T.zeros_like(X_flat), T.ones_like(X_flat))
        return nll.reshape(X.shape)

class MlpBernoulli(Layer):

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

    def sample(self):
        noise = rng.uniform(size=self.output.shape)
        sample = noise < self.output
        return sample

    def nll(self, X, inpt=None):
        n_dim = 2
        return supervised_loss(
            X, self.output, bern_ces,
            coord_axis=n_dim - 1,
            imp_weight=imp_weight)['loss_coord_wise']
