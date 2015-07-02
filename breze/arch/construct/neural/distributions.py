# -*- coding: utf-8 -*-

import theano.tensor as T
import numpy as np

from breze.arch.construct.base import Layer
from breze.arch.construct.neural import Mlp, FastDropoutMlp

from breze.arch.util import lookup
from breze.arch.component import transfer as _transfer
from breze.arch.component.distributions import DiagGauss, Bernoulli

def concat_transfer(inpt, mean_transfer, var_transfer):
    f_mean_transfer = lookup(mean_transfer, _transfer)
    f_var_transfer = lookup(var_transfer, _transfer)

    half = inpt.shape[-1] // 2
    if inpt.ndim == 3:
        mean, var = inpt[:, :, :half], inpt[:, :, half:]
        res = T.concatenate([f_mean_transfer(mean), f_var_transfer(var)], axis=2)
    else:
        mean, var = inpt[:, :half], inpt[:, half:]
        res = T.concatenate([f_mean_transfer(mean), f_var_transfer(var)], axis=1)
    return res


class MlpDiagGauss(DiagGauss):

    def __init__(self, inpt, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer_mean='identity', out_transfer_var=lambda x: x**2+1e-5, declare=None, name=None, rng=None):

        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer_mean = out_transfer_mean
        self.out_transfer_var = out_transfer_var

        super(MlpDiagGauss, self).__init__(inpt, declare, name, rng)

    def _forward(self):
        self.mlp = Mlp(self.inpt, self.n_inpt, self.n_hiddens, self.n_output*2,
                 self.hidden_transfers, lambda x: concat_transfer(x,self.out_transfer_mean,self.out_transfer_var), declare=self.declare)

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
                 hidden_transfers, out_transfer='sigmoid', declare=None, name=None, rng=None):
        if rng is None:
            self.rng = T.shared_randomstreams.RandomStreams()
        else:
            self.rng = rng

        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.out_transfer = out_transfer
        self.hidden_transfers = hidden_transfers

        super(MlpBernoulli, self).__init__(declare, name)

    def _forward(self):
        self.mlp = Mlp(self.inpt, self.n_inpt, self.n_hiddens, self.n_output,
                 self.hidden_transfers, self.out_transfer, declare=self.declare)

        self.layers = self.mlp.layers
        self.output = self.mlp.output
