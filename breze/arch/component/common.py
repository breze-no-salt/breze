# -*- coding: utf-8 -*-


import theano
import theano.tensor as T

import loss as loss_
from ...util import lookup, get_named_variables


def supervised_loss(target, prediction, loss, coord_axis=1):
    f_loss = lookup(loss, loss_)
    loss_coord_wise = f_loss(target, prediction)
    loss_sample_wise = loss_coord_wise.sum(axis=coord_axis)
    loss = loss_sample_wise.mean()

    return get_named_variables(locals())


def unsupervised_loss(prediction, loss, coord_axis=1):
    f_loss = lookup(loss, loss_)
    loss_coord_wise = f_loss(prediction)
    loss_sample_wise = loss_coord_wise.sum(axis=coord_axis)
    loss = loss_sample_wise.mean()

    return get_named_variables(locals())
