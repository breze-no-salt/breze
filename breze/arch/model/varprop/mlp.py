# -*- coding: utf-8 -*-


import theano.tensor as T

from ...util import lookup
from ...component.varprop import transfer
from ...model.neural import mlp


# TODO document


def mean_var_forward(in_mean, in_var, weights, bias, variance_bias_sqrt,
                     f, p_dropout):
    p_keep = 1 - p_dropout
    out_in_mean = T.dot(in_mean, weights) * p_keep
    out_in_mean += bias

    dropout_var = p_dropout * (1 - p_dropout)

    element_var = (dropout_var * in_var
                   + in_mean ** 2 * dropout_var
                   + in_var * p_keep ** 2)

    out_in_var = T.dot(element_var, weights ** 2)

    out_in_var *= variance_bias_sqrt ** 2
    out_mean, out_var = f(out_in_mean, out_in_var)
    return out_in_mean, out_in_var, out_mean, out_var


def int_mean_var_forward(in_mean, in_var, weights, bias, variance_bias_sqrt,
                         f, p_dropout):
    # TODO: document
    # in_mean: (N,), integers -- indexes the input matrix
    # in_var: (N,), floats -- represents the variance of the input
    in_mean = T.cast(in_mean, 'uint32')
    p_keep = 1 - p_dropout
    dropout_var = p_dropout * (1 - p_dropout)

    k_weight = weights[in_mean]
    k_weight.name = 'k_weight'

    out_in_mean = p_keep * k_weight + bias
    out_in_mean.name = 'out_in_mean'
    out_in_var = (dropout_var * weights[in_mean] ** 2
                  + p_keep * in_var.dimshuffle(0, 'x'))
    out_in_var.name = 'out_in_var'

    out_in_var *= variance_bias_sqrt ** 2
    out_mean, out_var = f(out_in_mean, out_in_var)
    return out_in_mean, out_in_var, out_mean, out_var


def parameters(n_inpt, n_hiddens, n_output, var_scale=True):
        spec = mlp.parameters(n_inpt, n_hiddens, n_output)

        if var_scale:
            for i, h in enumerate(n_hiddens):
                spec['hidden_var_scale%i' % (i + 1)] = h

            spec['out_var_scale'] = n_output

        return spec


def exprs(inpt_mean, inpt_var, target, in_to_hidden,
          hidden_to_hiddens,
          hidden_to_out,
          hidden_biases,
          hidden_var_scales,
          out_bias,
          out_var_scale,
          hidden_transfers, output_transfer,
          p_dropout_inpt,
          p_dropout_hiddens):
    exprs = {}

    f_hidden = lookup(hidden_transfers[0], transfer)
    hidden = mean_var_forward(inpt_mean, inpt_var, in_to_hidden,
                              hidden_biases[0], hidden_var_scales[0],
                              f_hidden, p_dropout_inpt)
    (hidden_in_mean, hidden_in_var, hidden_mean, hidden_var) = hidden

    exprs['hidden_in_mean_0'] = hidden_in_mean
    exprs['hidden_in_var_0'] = hidden_in_var
    exprs['hidden_mean_0'] = hidden_mean
    exprs['hidden_var_0'] = hidden_var

    zipped = zip(hidden_to_hiddens,
                 hidden_biases[1:],
                 hidden_var_scales[1:],
                 hidden_transfers[1:],
                 p_dropout_hiddens)

    for i, (w, b, bv, t, d) in enumerate(zipped):
        hidden_mean_m1, hidden_var_m1 = hidden_mean, hidden_var
        f = lookup(t, transfer)
        hidden = mean_var_forward(hidden_mean_m1, hidden_var_m1, w, b, bv, f, d)
        (hidden_in_mean, hidden_in_var, hidden_mean, hidden_var) = hidden

        exprs['hidden_in_mean_%i' % (i + 1)] = hidden_in_mean
        exprs['hidden_in_var_%i' % (i + 1)] = hidden_in_var
        exprs['hidden_mean_%i' % (i + 1)] = hidden_mean
        exprs['hidden_var_%i' % (i + 1)] = hidden_var

    f_output = lookup(output_transfer, transfer)
    output = mean_var_forward(hidden_mean, hidden_var, hidden_to_out,
                              out_bias, out_var_scale,
                              f_output, p_dropout_hiddens[-1])
    (output_in_mean, output_in_var, output_mean, output_var) = output
    output = T.concatenate([output_mean, output_var], axis=1)

    exprs.update({
        'inpt_mean': inpt_mean,
        'inpt_var': inpt_var,
        'target': target,
        'output_in_mean': output_in_mean,
        'output_in_var': output_in_var,
        'output_mean': output_in_mean,
        'output_var': output_in_var,
        'output': output,
    })

    return exprs
