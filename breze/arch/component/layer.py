# -*- coding: utf-8 -*-


import theano
import theano.tensor as T

from ..util import lookup, get_named_variables

import transfer
import corrupt


def simple(inpt, weights, bias, out_transfer, p_dropout=0, prefix=''):
    output_in = T.dot(inpt, weights) + bias
    f_output = lookup(out_transfer, transfer)
    output = f_output(output_in)
    if p_dropout != 0:
        output = corrupt.mask(output, p_dropout)

    return get_named_variables(locals(), prefix=prefix)
