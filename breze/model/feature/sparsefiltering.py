# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm


class SparseFiltering(Model):

    def __init__(self, n_inpt, n_feature, transfer_func='identity'):
        self.n_inpt = n_inpt
        self.n_feature = n_feature
        self.transfer_func = transfer_func

        super(SparseFiltering, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(self.n_inpt, self.n_feature)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        self.exprs = self.make_exprs(
            self.transfer_func, T.matrix('inpt'), self.parameters.in_to_feature)

    @staticmethod
    def get_parameter_spec(n_inpt, n_feature):
        return dict(in_to_feature=(n_inpt, n_feature))

    @staticmethod
    def make_exprs(transfer_func, inpt, inpt_to_feature):
        feature_in = T.dot(inpt, inpt_to_feature)
        transfer_func = lookup(transfer_func, transfer)
        feature = transfer_func(feature_in)

        col_normalized = T.sqrt(
            norm.normalize(feature, lambda x: x**2, axis=0) + 1E-8)
        row_normalized = T.sqrt(
            norm.normalize(col_normalized, lambda x: x**2, axis=1) + 1E-8)

        loss_rowwise = row_normalized.sum(axis=1)
        loss = loss_rowwise.mean()

        return {
            'inpt': inpt,
            'feature_in': feature_in,
            'feature': feature,
            'col_normalized': col_normalized,
            'row_normalized': row_normalized,
            'loss_rowwise': loss_rowwise,
            'loss': loss
        }


