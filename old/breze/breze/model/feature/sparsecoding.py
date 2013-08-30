# -*- coding: utf-8 -*-


import theano.tensor as T

from ...util import ParameterSet, Model


class SparseCoding(Model):

    # TODO: rename to c_sparsity
    def __init__(self, n_inpt, n_feature, c_l1):
        self.n_inpt = n_inpt
        self.n_feature = n_feature
        self.c_l1 = c_l1

        super(SparseCoding, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(self.n_inpt, self.n_feature)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        self.exprs = self.make_exprs(
            T.matrix('inpt'), self.parameters.feature_to_in, self.c_l1)

    @staticmethod
    def get_parameter_spec(n_inpt, n_feature):
        return dict(feature_to_in=(n_feature, n_inpt))

    @staticmethod
    def make_exprs(inpt, feature_to_in, c_l1):
        feature_flat = T.vector('feature_flat')
        feature = feature_flat.reshape((inpt.shape[0], feature_to_in.shape[0]))

        reconstruction = T.dot(feature, feature_to_in)
        residual = inpt - reconstruction

        reconstruct_loss_rowwise = (residual**2).sum(axis=1)
        sparsity_loss_rowwise = T.sqrt((feature**2) + 1e-4).sum(axis=1)
        loss_rowwise = reconstruct_loss_rowwise + c_l1 * sparsity_loss_rowwise

        reconstruct_loss = reconstruct_loss_rowwise.mean()
        sparsity_loss = sparsity_loss_rowwise.mean()
        loss = loss_rowwise.mean()

        # TODO normalize/constrain columns of weight matrix.

        return {
            'inpt': inpt,
            'feature_flat': feature_flat,
            'feature': feature,
            'reconstruction': reconstruction,
            'residual': residual,

            'reconstruct_loss_rowwise': reconstruct_loss_rowwise,
            'sparsity_loss_rowwise': sparsity_loss_rowwise,
            'loss_rowwise': loss_rowwise,

            'reconstruct_loss': reconstruct_loss,
            'sparsity_loss': sparsity_loss,
            'loss': loss
        }
