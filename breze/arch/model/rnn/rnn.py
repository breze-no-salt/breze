# -*- coding: utf-8 -*-


import numpy as np

import theano
import theano.tensor as T
from theano.tensor.extra_ops import repeat

from ...util import lookup
from ...component import transfer
from pooling import pooling_layer


def parameters(n_inpt, n_hiddens, n_output, skip_to_out=False, prefix=''):
    spec = dict(in_to_hidden=(n_inpt, n_hiddens[0]),
                hidden_to_out=(n_hiddens[-1], n_output),
                hidden_bias_0=n_hiddens[0],
                out_bias=n_output)

    zipped = zip(n_hiddens[:-1], n_hiddens[1:])
    for i, (inlayer, outlayer) in enumerate(zipped):
        spec['hidden_to_hidden_%i' % i] = (inlayer, outlayer)

    if skip_to_out:
        spec['in_to_out'] = (n_inpt, n_output)

    for i, h in enumerate(n_hiddens):
        spec['hidden_bias_%i' % i] = h
        spec['recurrent_%i' % i] = (h, h)
        spec['initial_hiddens_%i' % i] = h
        if skip_to_out and i < len(n_hiddens):
            # Only do for all but the last layer.
            spec['hidden_%i_to_out' % i] = (h, n_output)

    spec = dict(('%s%s'% (prefix, k), v)  for k, v in spec.items())

    return spec


def recurrent_layer(hidden_inpt, hidden_to_hidden, f, initial_hidden):
    def step(x, hi_tm1):
        h_tm1 = f(hi_tm1)
        hi = T.dot(h_tm1, hidden_to_hidden) + x
        return hi

    # Modify the initial hidden state to obtain several copies of
    # it, one per sample.
    # TODO check if this is correct; FD-RNNs do it right.
    initial_hidden_b = repeat(initial_hidden, hidden_inpt.shape[1], axis=0)
    initial_hidden_b = initial_hidden_b.reshape(
        (hidden_inpt.shape[1], hidden_inpt.shape[2]))

    hidden_in_rec, _ = theano.scan(
        step,
        sequences=hidden_inpt,
        outputs_info=[initial_hidden_b])

    hidden_rec = f(hidden_in_rec)

    return hidden_in_rec, hidden_rec


def feedforward_layer(inpt, weights, bias):
    n_time_steps = inpt.shape[0]
    n_samples = inpt.shape[1]

    n_inpt = weights.shape[0]
    n_output = weights.shape[1]

    inpt_flat = inpt.reshape((n_time_steps * n_samples, n_inpt))
    output_flat = T.dot(inpt_flat, weights)
    output = output_flat.reshape((n_time_steps, n_samples, n_output))
    output += bias.dimshuffle('x', 'x', 0)
    return output


def leaky_integration(inpt, coefficients):
    def step(x, y_tm1):
        c = coefficients[np.newaxis]
        y = c * y_tm1 + (1 - c) * x
        return y
    output, _ = theano.scan(
        step,
        sequences=inpt,
        outputs_info=[T.zeros_like(inpt[0])])
    return output


def exprs(inpt, in_to_hidden, hidden_to_hiddens, hidden_to_out,
          hidden_biases, initial_hiddens, recurrents, out_bias,
          hidden_transfers, out_transfer, pooling=None, leaky_coeffs=None,
          in_to_out=None, skip_to_outs=None):
    exprs = {}

    f_hiddens = [lookup(i, transfer) for i in hidden_transfers]
    f_output = lookup(out_transfer, transfer)

    hidden_in = feedforward_layer(inpt, in_to_hidden, hidden_biases[0])
    hidden_in_rec, hidden_rec = recurrent_layer(
        hidden_in, recurrents[0], f_hiddens[0], initial_hiddens[0])
    exprs['hidden_in_0'] = hidden_in_rec
    if leaky_coeffs is not None:
        hidden_rec = leaky_integration(hidden_rec, leaky_coeffs[0])
    exprs['hidden_0'] = hidden_rec

    zipped = zip(hidden_to_hiddens, hidden_biases[1:], recurrents[1:],
                 f_hiddens[1:], initial_hiddens[1:])

    for i, (w, b, r, t, j) in enumerate(zipped):
        hidden_m1 = hidden_rec
        hidden_in = feedforward_layer(hidden_m1, w, b)
        hidden_in_rec, hidden_rec = recurrent_layer(hidden_in, r, t, j)
        if leaky_coeffs is not None:
            hidden_rec = leaky_integration(hidden_rec, leaky_coeffs[i])
        exprs['hidden_in_%i' % (i + 1)] = hidden_in_rec
        exprs['hidden_%i' % (i + 1)] = hidden_rec

    unpooled = feedforward_layer(hidden_rec, hidden_to_out, out_bias)

    if in_to_out is not None:
        unpooled += feedforward_layer(inpt, in_to_out, T.zeros_like(out_bias))

    if skip_to_outs is not None:
        for i, s in enumerate(skip_to_outs):
            unpooled += feedforward_layer(
                exprs['hidden_%i' % i], s, T.zeros_like(out_bias))

    if pooling is None:
        output_in = unpooled
    else:
        output_in = pooling_layer(unpooled, pooling)

    output = f_output(output_in)

    exprs.update(
        {'inpt': inpt,
         'unpooled': unpooled,
         'output_in': output_in,
         'output': output,
         })

    return exprs
