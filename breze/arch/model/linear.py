# -*- coding: utf-8 -*-


from ..util import lookup, get_named_variables
from ..component import loss as loss_, layer


def parameters(n_inpt, n_output):
    return {'in_to_out': (n_inpt, n_output),
            'bias': n_output}


def exprs(inpt, weights, bias, out_transfer):
    exprs = layer.simple(inpt, weights, bias, out_transfer)
    output_in, output = exprs['output_in'], exprs['output']

    return get_named_variables(locals())
