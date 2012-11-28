# -*- coding: utf-8 -*-


import theano.tensor as T

from ...util import ParameterSet, lookup
from ...component import distance
from ..neural import TwoLayerPerceptron


class AutoEncoder(TwoLayerPerceptron):

    def __init__(self, n_inpt, n_hidden,
                 hidden_transfer, out_transfer, loss,
                 tied_weights=True):
        self.tied_weights = tied_weights
        self.feature_transfer = hidden_transfer
        self.n_feature = n_hidden

        super(AutoEncoder, self).__init__(
            n_inpt, n_hidden, n_inpt,
            hidden_transfer, out_transfer, loss)

    def init_pars(self):
        parspec = self.get_parameter_spec(
            self.n_inpt, self.n_hidden, self.tied_weights)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        if self.tied_weights:
            hidden_to_out = self.parameters.in_to_hidden.T
        else:
            hidden_to_out = self.parameters.hidden_to_out

        self.exprs = self.make_exprs(
            T.matrix('inpt'),
            self.parameters.in_to_hidden, hidden_to_out,
            self.parameters.hidden_bias, self.parameters.out_bias,
            self.hidden_transfer, self.out_transfer, self.loss)
        self.exprs['feature'] = self.exprs['hidden']

    @staticmethod
    def get_parameter_spec(n_inpt, n_hidden, tied_weights):
        if tied_weights:
            return dict(in_to_hidden=(n_inpt, n_hidden),
                        hidden_bias=n_hidden,
                        out_bias=n_inpt)
        else:
            return dict(in_to_hidden=(n_inpt, n_hidden),
                        hidden_to_out=(n_hidden, n_inpt),
                        hidden_bias=n_hidden,
                        out_bias=n_inpt)

    @staticmethod
    def make_exprs(inpt, in_to_hidden, hidden_to_out,
                   hidden_bias, out_bias,
                   hidden_transfer, out_transfer, loss):
        return TwoLayerPerceptron.make_exprs(
            inpt, inpt, in_to_hidden, hidden_to_out,
            hidden_bias, out_bias,
            hidden_transfer, out_transfer, loss)


class DenoisingAutoEncoder(AutoEncoder):

    def __init__(self, n_inpt, n_hidden, hidden_transfer, out_transfer,
                 loss, tied_weights=True):
        super(DenoisingAutoEncoder, self).__init__(
            n_inpt, n_hidden,
            hidden_transfer, out_transfer, loss, tied_weights)

    @staticmethod
    def make_exprs(inpt, in_to_hidden, hidden_to_out,
                   hidden_bias, out_bias,
                   hidden_transfer, out_transfer, loss):
        corrupted = T.matrix('corrupted')
        exprs = TwoLayerPerceptron.make_exprs(
            corrupted, inpt, in_to_hidden, hidden_to_out,
            hidden_bias, out_bias, hidden_transfer, out_transfer,
            loss)
        exprs["corrupted"] = exprs["inpt"]
        exprs["inpt"] = exprs["target"]
        return exprs


class SparseAutoEncoder(AutoEncoder):

    def __init__(self, n_inpt, n_hidden, hidden_transfer, out_transfer,
                 loss,
                 c_sparsity, sparsity_loss, sparsity_target=0.01,
                 tied_weights=True):
        self.c_sparsity = c_sparsity
        self.sparsity_loss = sparsity_loss
        self.sparsity_target = sparsity_target

        super(SparseAutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer,
            loss, tied_weights)

    def init_exprs(self):
        if self.tied_weights:
            hidden_to_out = self.parameters.in_to_hidden.T
        else:
            hidden_to_out = self.parameters.hidden_to_out

        self.exprs = self.make_exprs(
            T.matrix('inpt'), self.parameters.in_to_hidden, hidden_to_out,
            self.parameters.hidden_bias, self.parameters.out_bias,
            self.hidden_transfer, self.out_transfer, self.loss,
            self.sparsity_loss, self.c_sparsity, self.sparsity_target)

    @staticmethod
    def make_exprs(inpt, in_to_hidden, hidden_to_out,
                   hidden_bias, out_bias,
                   hidden_transfer, out_transfer, loss,
                   sparsity_loss, c_sparsity, sparsity_target):
        exprs = AutoEncoder.make_exprs(
            inpt, in_to_hidden, hidden_to_out,
            hidden_bias, out_bias,
            hidden_transfer, out_transfer, loss)

        hidden = exprs['hidden']
        f_distance = lookup(sparsity_loss, distance)

        sparsity_loss = f_distance(sparsity_target, hidden.mean(axis=0))

        exprs['sparsity_loss'] = sparsity_loss
        exprs['reconstruct_loss'] = exprs['loss']
        exprs['loss'] = exprs['reconstruct_loss'] + c_sparsity * sparsity_loss

        return exprs


class ContractiveAutoEncoder(AutoEncoder):

    def __init__(self, n_inpt, n_hidden, hidden_transfer, out_transfer,
                 loss, c_jacobian, tied_weights=True):

        self.c_jacobian = c_jacobian

        super(ContractiveAutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer,
            loss, tied_weights)

    def init_exprs(self):
        if self.tied_weights:
            hidden_to_out = self.parameters.in_to_hidden.T
        else:
            hidden_to_out = self.parameters.hidden_to_out

        self.exprs = self.make_exprs(
            T.matrix('inpt'), self.parameters.in_to_hidden, hidden_to_out,
            self.parameters.hidden_bias, self.parameters.out_bias,
            self.hidden_transfer, self.out_transfer, self.loss,
            self.c_jacobian)

    @staticmethod
    def make_exprs(inpt, in_to_hidden, hidden_to_out,
                   hidden_bias, out_bias,
                   hidden_transfer, out_transfer, loss,
                   c_jacobian):
        exprs = AutoEncoder.make_exprs(
            inpt, in_to_hidden, hidden_to_out,
            hidden_bias, out_bias,
            hidden_transfer, out_transfer, loss)
        hidden = exprs['hidden']
        hidden_in = exprs['hidden_in']

        d_h_d_h_in = T.grad(hidden.sum(), hidden_in)
        jacobian_loss = T.sum(
            T.mean(d_h_d_h_in**2, axis=0) * (in_to_hidden**2))

        exprs['jacobian_loss'] = jacobian_loss
        exprs['reconstruct_loss'] = exprs['loss']
        exprs['loss'] = exprs['reconstruct_loss'] + c_jacobian * jacobian_loss

        return exprs
