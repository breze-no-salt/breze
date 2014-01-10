# -*- coding: utf-8 -*-


import theano.tensor as T

from ...util import ParameterSet, Model, lookup
from ...component import transfer as _transfer, loss as loss_
from breze.arch.util import ParameterSet, Model, lookup, get_named_variables


def ica_loss(code, transfer):
    f_transfer= lookup(transfer, _transfer)
    ica_loss_coord_wise = f_transfer(code)
    ica_loss_sample_wise = ica_loss_coord_wise.sum(axis=1)
    ica_loss = ica_loss_sample_wise.mean()
    return get_named_variables(locals())
