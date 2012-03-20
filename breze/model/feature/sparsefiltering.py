# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance, norm


def make_exprs(transfer_func, inpt, inpt_to_feature):
    feature_in = T.dot(inpt, inpt_to_feature)
    transfer_func = lookup(transfer_func, transfer)
    feature = transfer_func(feature_in)

    col_normalized = norm.normalize(feature, norm.root_l2, axis=0)
    row_normalized = norm.normalize(col_normalized, norm.root_l2, axis=1)

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


class SparseFiltering(Model):

    def __init__(self, n_inpt, n_features, transfer_func='identity'):
        self.n_inpt = n_inpt
        self.n_features = n_features
        self.transfer = transfer

        pars = self.make_pars()

        exprs = make_exprs(transfer_func, T.matrix('inpt'),
                           pars.inpt_to_feature)

        super(SparseFiltering, self).__init__(exprs, pars)

    def make_pars(self):
        return ParameterSet(inpt_to_feature=(self.n_inpt, self.n_features))
