# -*- coding: utf-8 -*-


import theano.tensor as T

from ...util import ParameterSet, Model, lookup
from ...component import transfer, loss as loss_


class Rica(Model):

    def __init__(self, n_inpt, n_hidden, hidden_transfer, feature_transfer,
                 out_transfer, loss, c_ica):
        self.n_inpt = n_inpt
        self.n_hidden = n_hidden
        self.hidden_transfer = hidden_transfer
        self.feature_transfer = feature_transfer
        self.out_transfer = out_transfer
        self.loss = loss
        self.c_ica = c_ica

        super(Rica, self).__init__()

    def init_exprs(self):
        self.exprs = self.make_exprs(
            T.matrix('inpt'), self.parameters.in_to_hidden,
            self.hidden_transfer, self.feature_transfer, self.out_transfer,
            self.loss, self.c_ica)

    def init_pars(self):
        parspec = self.get_parameter_spec(self.n_inpt, self.n_hidden)
        self.parameters = ParameterSet(**parspec)

    @staticmethod
    def get_parameter_spec(n_inpt, n_hidden):
        return dict(in_to_hidden=(n_inpt, n_hidden))

    @staticmethod
    def make_exprs(inpt, in_to_hidden, hidden_transfer, feature_transfer,
                   out_transfer, loss, c_ica):
        f_hidden = lookup(hidden_transfer, transfer)
        f_feature = lookup(feature_transfer, transfer)
        f_output = lookup(out_transfer, transfer)
        f_loss = lookup(loss, loss_)

        in_to_hidden_normed = in_to_hidden / T.sqrt((in_to_hidden**2).sum(axis=0)).dimshuffle('x', 0)

        hidden_in = T.dot(inpt, in_to_hidden_normed)
        hidden = f_hidden(hidden_in)

        output_in = T.dot(hidden, in_to_hidden_normed.T)
        output = f_output(output_in)

        feature = f_feature(hidden)

        recons_loss_rowwise = f_loss(inpt, output).sum(axis=1)
        ica_loss_rowwise = feature.sum(axis=1)

        loss_rowwise = recons_loss_rowwise + c_ica * ica_loss_rowwise
        loss = loss_rowwise.mean()
        recons_loss = recons_loss_rowwise.mean()
        ica_loss = ica_loss_rowwise.mean()

        exprs = {
            'inpt': inpt,
            'in_to_hidden_normed': in_to_hidden_normed,
            'feature': feature,
            'hidden_in': hidden_in,
            'hidden': hidden,
            'reconstruct_loss': recons_loss,
            'output_in': output_in,
            'output': output,
            'ica_loss': ica_loss,
            'loss': loss,
            'loss_rowwise': loss_rowwise,
            'ica_loss': ica_loss,
            'ica_loss_rowwise': ica_loss_rowwise,
            'recons_loss': recons_loss,
            'recons_loss_rowwise': recons_loss_rowwise,
        }

        return exprs
