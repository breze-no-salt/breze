__author__ = 'apuigdom'
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.extra_ops import repeat
from theano.tensor.signal import downsample

from ...util import lookup
from ...component import transfer, loss as loss_
import numpy as np

import mlp

# TODO document

def parameters(n_inpt, n_hidden_conv, n_hidden_full, n_output,
               filter_shapes, image_shapes, recurrency):
    spec = dict(in_to_hidden=(n_hidden_conv[0], n_inpt[1],
                filter_shapes[0][0], filter_shapes[0][1]),
                hidden_to_out=(n_hidden_full[-1], n_output),
                hidden_conv_to_hidden_full=(n_hidden_conv[-1] * np.prod(image_shapes[-1][-2:]),
                                            n_hidden_full[0]),
                hidden_conv_bias_0=n_hidden_conv[0],
                hidden_full_bias_0=n_hidden_full[0],
                out_bias=n_output)
    if recurrency[len(n_hidden_conv)-1]:
        size_hidden = n_hidden_conv[-1] * np.prod(image_shapes[-1][-2:])
        spec['recurrent_%i' % len(n_hidden_conv)] = (size_hidden, size_hidden)
        spec['initial_hiddens_%i' % len(n_hidden_conv)] = size_hidden
    zipped = zip(n_hidden_conv[:-1], n_hidden_conv[1:], filter_shapes[1:],
                 image_shapes[:-1], recurrency[:len(n_hidden_conv)])
    for i, (inlayer, outlayer, filter_shape, image_shape, rec) in enumerate(zipped):
        spec['hidden_conv_to_hidden_conv_%i' % i] = (
            outlayer, inlayer, filter_shape[0], filter_shape[1])
        spec['hidden_conv_bias_%i' % (i + 1)] = outlayer
        if rec:
            size_hidden = inlayer * np.prod(image_shape[-2:])
            spec['recurrent_%i' % i] = (size_hidden, size_hidden)
            spec['initial_hiddens_%i' % i] = size_hidden

    zipped = zip(n_hidden_full[:-1], n_hidden_full[1:], recurrency[len(n_hidden_conv):])
    for i, (inlayer, outlayer, rec) in enumerate(zipped):
        spec['hidden_full_to_hidden_full_%i' % i] = (inlayer, outlayer)
        spec['hidden_full_bias_%i' % (i + 1)] = outlayer
        if rec:
            size_hidden = inlayer * np.prod(image_shape[-2:])
            spec['recurrent_%i' % i] = (size_hidden, size_hidden)
            spec['initial_hiddens_%i' % i] = size_hidden
    return spec


def recurrent_layer(hidden_inpt, hidden_to_hidden, f, initial_hidden):
    def step_full(x, hi_tm1):
        h_tm1 = f(hi_tm1)
        hi = T.dot(h_tm1, hidden_to_hidden) + x
        return hi
    def step_conv(x, hi_tm1):
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
        step_full,
        sequences=hidden_inpt,
        outputs_info=[initial_hidden_b])

    return hidden_in_rec


def conv_part(inpt, params):
    w, b, fs, ims, ps = params
    hidden_in = conv.conv2d(inpt, w, filter_shape=fs,
                            image_shape=ims)
    hidden_in_predown = downsample.max_pool_2d(
        hidden_in, ps, ignore_border=True)
    hidden_in_down = hidden_in_predown + b.dimshuffle('x', 0, 'x', 'x')
    return hidden_in_down


def feedforward_layer(inpt, weights, bias):
    n_time_steps = inpt.shape[0]
    n_samples = inpt.shape[1]

    n_inpt = weights.shape[0]
    n_output = weights.shape[1]

    inpt_flat = inpt.reshape((n_time_steps * n_samples, n_inpt))
    output_flat = T.dot(inpt_flat, weight)
    output = output_flat.reshape((n_time_steps, n_samples, n_output))
    output += bias.dimshuffle('x', 'x', 0)
    return output


def exprs(inpt, target, in_to_hidden, hidden_to_out, out_bias,
          hidden_conv_to_hidden_full, hidden_conv_to_hidden_conv,
          hidden_full_to_hidden_full, hidden_conv_bias,
          hidden_full_bias, hidden_conv_transfers,
          hidden_full_transfers, output_transfer,
          image_shapes, filter_shapes_comp, input_shape, pool_shapes,
          recurrents, initial_hiddens, recurrent_flags):

    #input shape = n_time_steps, n_samples, channels, n_frames_to_take, n_output
    #conv part: reshape to n_time_step * n_samples, channels, n_frames_to_take, n_output
    #rec part: reshape to n_time_steps, n_samples, channels * n_frames_to_take * n_output
    exprs = {}

    f_hidden = lookup(hidden_conv_transfers[0], transfer)

    # Convolutional part
    zipped = zip(input_shapes[1:], recurrent_flags, hidden_conv_transfers,
                 recurrents, initial_hiddens,
                 [in_to_hidden] + hidden_conv_to_hidden_conv, hidden_conv_bias,
                 filter_shapes_comp, image_shapes, pool_shapes)

    conv_shape = [np.prod(input_shape[:2])] + input_shape[2:]
    reshaped_inpt = inpt.reshape(conv_shape)
    hidden = inpt
    current_rec = 0
    for i, params in enumerate(zipped):
        inpt_shape, recurrent_flag, ft, rec, ih = params[:3]
        f = lookup(ft, transfer)
        hidden_in_down = conv_part(hidden, params[3:])
        if recurrent_flag:
            rec_shape = input_shape[:2] + [np.prod(input_shape[2:])]
            conv_shape = [np.prod(input_shape[:2])] + input_shape[2:]
            reshaped_hidden_in_conv_down = hidden_in_conv_down.reshape(inpt_shape).reshape(rec_shape)
            hidden_in_rec = recurrent_layer(reshaped_hidden_in_conv_down, rec, f, ih)
            hidden_in_down = hidden_in_rec.reshape(input_shape).reshape(conv_shape)
        exprs['conv-hidden_in_%i' % i] = hidden_in_down
        hidden = exprs['conv-hidden_%i' % i] = f(hidden_in_down)

    # Mlp part
    offset = len(hidden_conv_to_hidden_conv)+1
    zipped = zip([hidden_conv_to_hidden_full] + hidden_full_to_hidden_full,
                 hidden_full_bias, hidden_full_transfers, recurrents[offset:],
                 initial_hiddens[offset:], recurrent_flags[offset:])

    hidden = hidden.reshape(inpt_shape[offset])
    for i, (w, b, t, rec, ih, rf) in enumerate(zipped):
        hidden_in = feedforward_layer(hidden, w, b)
        f = lookup(t, transfer)
        if rf:
            hidden_in = recurrent_layer(hidden_in, rec, f, ih)
        hidden = f(hidden_in)
        exprs['hidden_in_%i' % (i + offset)] = hidden_in
        exprs['hidden_%i' % (i + offset)] = hidden

    f_output = lookup(output_transfer, transfer)


    output_in = feedforward_layer(hidden, hidden_to_out, out_bias)

    output = f_output(output_in)

    exprs.update({
        'inpt': inpt,
        'target': target,
        'output_in': output_in,
        'output': output
    })

    return exprs
