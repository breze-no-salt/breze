# -*- coding: utf-8 -*-


import theano.tensor as T

from ...util import lookup, get_named_variables
from ...component import transfer as _transfer, norm


def parameters(n_inpt, n_output):
    return dict(in_to_out=(n_inpt, n_output))


def loss(output, transfer):
    f_transfer = lookup(transfer, _transfer)
    output_post = f_transfer(output)

    col_normalized = T.sqrt(
        norm.normalize(output_post, lambda x: x ** 2, axis=0) + 1E-8)
    row_normalized = T.sqrt(
        norm.normalize(col_normalized, lambda x: x ** 2, axis=1) + 1E-8)

    loss_rowwise = row_normalized.sum(axis=1)
    loss = loss_rowwise.mean()

    return get_named_variables(locals())
