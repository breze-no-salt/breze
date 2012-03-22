# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm

from autoencoder import AutoEncoder


class Rica(AutoEncoder):

    def __init__(self, n_inpt, n_hidden, 
            feature_transfer, out_transfer,
            reconstruct_loss, 
            c_ica, 
            tied_weights=True):

        self.feature_transfer = feature_transfer
        self.c_ica = c_ica
        super(Rica, self).__init__(
            n_inpt, n_hidden, 'identity', out_transfer,
            reconstruct_loss, tied_weights)

    def init_exprs(self):
        if self.tied_weights:
            hidden_to_output = self.parameters.inpt_to_hidden.T
        else:
            hidden_to_output = self.parameters.hidden_to_output

        self.exprs = self.make_exprs(
            T.matrix('inpt'), self.parameters.inpt_to_hidden, hidden_to_output,
            self.parameters.hidden_bias, self.parameters.out_bias,
            self.hidden_transfer, self.feature_transfer, self.out_transfer, 
            self.reconstruct_loss,
            self.c_ica)

    @staticmethod
    def make_exprs(inpt, inpt_to_hidden, hidden_to_output, 
                   hidden_bias, out_bias,
                   hidden_transfer, feature_transfer, out_transfer,
                   reconstruct_loss, c_ica):

        inpt_to_hidden_normed = T.sqrt(
            norm.normalize(inpt_to_hidden, lambda x: x**2, axis=0))
        hidden_to_output_normed = T.sqrt(
                norm.normalize(hidden_to_output, lambda x: x**2, axis=1))

        exprs = AutoEncoder.make_exprs(
            inpt, inpt_to_hidden_normed, hidden_to_output_normed, 
            hidden_bias, out_bias, 
            hidden_transfer, out_transfer, reconstruct_loss)

        f_feature = lookup(feature_transfer, transfer)

        exprs['reconstruct_loss'] = exprs['loss']

        exprs['feature'] = f_feature(exprs['hidden'])
        exprs['ica_loss'] = exprs['feature'].mean()

        exprs['loss'] = exprs['reconstruct_loss'] + c_ica * exprs['ica_loss']
        return exprs
