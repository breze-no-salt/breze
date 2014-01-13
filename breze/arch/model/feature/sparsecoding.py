# -*- coding: utf-8 -*-


import theano.tensor as T

from ...util import get_named_variables


def parameters(n_feature, n_inpt):
    return dict(feature_to_in=(n_feature, n_inpt))


def exprs(inpt, feature_to_in, c_sparsity):
    feature_flat = T.vector('feature_flat')
    feature = feature_flat.reshape((inpt.shape[0], feature_to_in.shape[0]))

    reconstruction = T.dot(feature, feature_to_in)
    residual = inpt - reconstruction

    rec_loss_coord_wise = residual ** 2
    rec_loss_sample_wise = rec_loss_coord_wise.sum(axis=1)
    rec_loss = rec_loss_sample_wise.mean()

    sparsity_loss_coord_wise = T.sqrt((feature ** 2) + 1e-4)
    sparsity_loss_sample_wise = sparsity_loss_coord_wise.sum(axis=1)
    sparsity_loss = sparsity_loss_sample_wise.mean()

    loss = rec_loss + c_sparsity * sparsity_loss

    return get_named_variables(locals())
