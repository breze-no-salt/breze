# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm


class AutoEncoder(Model):

    def __init__(self, n_inpt, n_hidden, 
                 hidden_transfer, out_transfer, reconstruct_loss,
                 tied_weights=True):
        self.n_inpt = n_inpt
        self.n_hidden = n_hidden
        self.tied_weights = tied_weights

        self.hidden_transfer = hidden_transfer
        self.out_transfer = out_transfer
        self.reconstruct_loss = reconstruct_loss

        self.init_pars()
        self.init_exprs()

    def init_pars(self):
        if self.tied_weights:
            self.parameters = ParameterSet(
                inpt_to_hidden=(self.n_inpt, self.n_hidden),
                hidden_bias=self.n_hidden,
                out_bias=self.n_inpt)
        else:
            self.parameters = ParameterSet(
                inpt_to_hidden=(self.n_inpt, self.n_hidden),
                hidden_to_output=(self.n_hidden, self.n_inpt),
                hidden_bias=self.n_hidden,
                out_bias=self.n_inpt)

    def init_exprs(self):
        if self.tied_weights:
            hidden_to_outpt = self.parameters.inpt_to_hidden.T
        else:
            hidden_to_outpt = self.parameters.hidden_to_output

        self.exprs = self.make_exprs(
            T.matrix('inpt'), self.parameters.inpt_to_hidden, hidden_to_output,
            self.parameters.hidden, self.parameters.bias_out,
            self.hidden_transfer, self.out_transfer, self.reconstruct_loss)

    @staticmethod
    def make_exprs(inpt, inpt_to_hidden, hidden_to_output, 
                   hidden_bias, out_bias,
                   hidden_transfer, out_transfer, reconstruct_loss):

        f_hidden = lookup(hidden_transfer, transfer)
        f_out = lookup(out_transfer, transfer)
        f_reconstruct_loss = lookup(reconstruct_loss, distance)

        # Define model.
        hidden_in = T.dot(inpt, inpt_to_hidden) + hidden_bias
        hidden = f_hidden(hidden_in)
        output_in = T.dot(hidden, hidden_to_output) + out_bias
        output = f_out(output_in)

        # Define loss.
        loss_rowwise = f_reconstruct_loss(inpt, output, axis=1)
        loss = loss_rowwise.mean()

        return {'inpt': inpt,
                'hidden_in': hidden_in,
                'hidden': hidden,
                'output_in': output_in,
                'output': output,
                'loss_rowwise': loss_rowwise,
                'loss': loss}


class SparseAutoEncoder(AutoEncoder):

    def __init__(self, n_inpt, n_hidden, hidden_transfer, out_transfer,
            reconstruct_loss, 
            sparsity_loss, sparsity_penalty, sparsity_target=0.01,
            tied_weights=True):
        self.sparsity_penalty = sparsity_penalty
        self.sparsity_loss = sparsity_loss
        self.sparsity_target = sparsity_target

        super(SparseAutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer,
            reconstruct_loss, tied_weights)

    def init_exprs(self):
        if self.tied_weights:
            hidden_to_output = self.parameters.inpt_to_hidden.T
        else:
            hidden_to_output = self.parameters.hidden_to_output

        self.exprs = self.make_exprs(
            T.matrix('inpt'), self.parameters.inpt_to_hidden, hidden_to_output,
            self.parameters.hidden_bias, self.parameters.out_bias,
            self.hidden_transfer, self.out_transfer, self.reconstruct_loss,
            self.sparsity_loss, self.sparsity_penalty, self.sparsity_target)

    @staticmethod
    def make_exprs(inpt, inpt_to_hidden, hidden_to_output,
                   hidden_bias, out_bias,
                   hidden_transfer, out_transfer, reconstruct_loss,
                   sparsity_loss, sparsity_penalty, sparsity_target):
        exprs = AutoEncoder.make_exprs(
            inpt, inpt_to_hidden, hidden_to_output, 
            hidden_bias, out_bias, 
            hidden_transfer, out_transfer, reconstruct_loss)

        hidden = exprs['hidden']
        f_distance = lookup(sparsity_loss, distance)

        sparsity_loss = f_distance(sparsity_target, hidden.mean(axis=0))
        sparsity_loss *= sparsity_penalty

        exprs['sparsity_loss'] = sparsity_loss
        exprs['loss_reg'] = exprs['loss'] + sparsity_loss
        
        return exprs


class ContractiveAutoEncoder(AutoEncoder):

    def __init__(self, n_inpt, n_hidden, hidden_transfer, out_transfer,
                 reconstruct_loss, jacobian_penalty, tied_weights=True):

        self.jacobian_penalty = jacobian_penalty
        
        super(ContractiveAutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer,
            reconstruct_loss, tied_weights)

    def init_exprs(self):
        if self.tied_weights:
            hidden_to_output = self.parameters.inpt_to_hidden.T
        else:
            hidden_to_output = self.parameters.hidden_to_output

        self.exprs = self.make_exprs(
            T.matrix('inpt'), self.parameters.inpt_to_hidden, hidden_to_output,
            self.parameters.hidden_bias, self.parameters.out_bias,
            self.hidden_transfer, self.out_transfer, self.reconstruct_loss,
            self.jacobian_penalty)

    @staticmethod
    def make_exprs(inpt, inpt_to_hidden, hidden_to_output,
                   hidden_bias, out_bias,
                   hidden_transfer, out_transfer, reconstruct_loss,
                   jacobian_penalty):
        exprs = AutoEncoder.make_exprs(
            inpt, inpt_to_hidden, hidden_to_output, 
            hidden_bias, out_bias, 
            hidden_transfer, out_transfer, reconstruct_loss)
        hidden = exprs['hidden']
        hidden_in = exprs['hidden_in']

        d_h_d_h_in = T.grad(hidden.sum(), hidden_in)
        #jacobian = (d_h_d_h_in.dimshuffle(0, 1, 'x') * 
        #            inpt_to_hidden.T)
        #jacobian_loss = (jacobian**2).mean(axis=0).sum() * jacobian_penalty
        jacobian_loss = T.sum(T.mean(d_h_d_h_in**2, axis=0) * (inpt_to_hidden**2))

        exprs['jacobian_loss'] = jacobian_loss
        exprs['loss_reg'] = exprs['loss'] + jacobian_loss
        
        return exprs
