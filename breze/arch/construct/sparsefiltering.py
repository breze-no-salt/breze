# -*- coding: utf-8 -*-

import theano.tensor as T

from breze.arch.component import transfer as _transfer, norm
from breze.arch.construct.base import Layer
from breze.arch.util import lookup


class SparseFilteringLoss(Layer):

    def __init__(self, inpt, density,
                 comp_dim=1, imp_weight=None, declare=None, name=None):
        self.inpt = inpt
        self.density = density

        super(SparseFilteringLoss, self).__init__(declare, name)

    def _forward(self):
        f_density = lookup(self.density, _transfer)
        output = f_density(self.inpt)

        col_normalized = T.sqrt(
            norm.normalize(output, lambda x: x ** 2, axis=0) + 1E-8)
        row_normalized = T.sqrt(
            norm.normalize(col_normalized, lambda x: x ** 2, axis=1) + 1E-8)

        loss_sample_wise = row_normalized.sum(axis=1)
        self.total = loss_sample_wise.mean()
