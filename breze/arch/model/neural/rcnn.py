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



def define_recurrent_params(spec, current_layer, size_hidden, recurrency):
    if recurrency == 'full':
        print size_hidden
        spec['recurrent_%i' % current_layer] = (size_hidden, size_hidden)
    elif recurrency == 'single':
        spec['recurrent_%i' % current_layer] = size_hidden
    elif recurrency[0] == 'double':
        spec['recurrent_%i_0' % current_layer] = (size_hidden, recurrency[1])
        spec['recurrent_%i_1' % current_layer] = (recurrency[1], size_hidden)
    elif recurrency[0] == 'conv':
        spec['recurrent_%i' % current_layer] = (1, 1, recurrency[1], 1)
    else:
        raise AttributeError('Recurrency format is not correct')
    return spec

def parameters(n_inpt, n_hidden_conv, n_hidden_full, n_output,
               filter_shapes, image_shapes, recurrency):
    spec = dict(in_to_hidden=(n_hidden_conv[0], n_inpt[2],
                filter_shapes[0][0], filter_shapes[0][1]),
                hidden_to_out=(n_hidden_full[-1], n_output),
                hidden_conv_bias_0=n_hidden_conv[0],
                hidden_full_bias_0=n_hidden_full[0],
                out_bias=n_output)

    if recurrency[0]:
        size_hidden = np.prod(image_shapes[1][-3:])
        spec = define_recurrent_params(spec, 0, size_hidden, recurrency[0])
        spec['initial_hiddens_0'] = size_hidden

    zipped = zip(n_hidden_conv[:-1], n_hidden_conv[1:], filter_shapes[1:],
                 image_shapes[2:], recurrency[1:len(n_hidden_conv)])
    for i, (inlayer, outlayer, filter_shape, image_shape, rec) in enumerate(zipped):
        spec['hidden_conv_to_hidden_conv_%i' % i] = (
            outlayer, inlayer, filter_shape[0], filter_shape[1])
        spec['hidden_conv_bias_%i' % (i + 1)] = outlayer
        if rec:
            size_hidden = np.prod(image_shape[-3:])
            spec = define_recurrent_params(spec, i+1, size_hidden, rec)
            spec['initial_hiddens_%i' % (i+1)] = size_hidden

    spec['hidden_conv_to_hidden_full'] = (np.prod(image_shapes[len(n_hidden_conv)][-3:]), n_hidden_full[0])
    current_layer = len(n_hidden_conv)+1
    size_hidden = image_shapes[current_layer][-1]
    if recurrency[current_layer-1]:
        spec = define_recurrent_params(spec, current_layer, size_hidden, recurrency[current_layer-1])
        spec['initial_hiddens_%i' % current_layer] = size_hidden


    zipped = zip(n_hidden_full[:-1], n_hidden_full[1:],
                 image_shapes[len(n_hidden_conv)+2:],
                 recurrency[len(n_hidden_conv)+1:])
    for i, (inlayer, outlayer, image_shape, rec) in enumerate(zipped):
        spec['hidden_full_to_hidden_full_%i' % i] = (inlayer, outlayer)
        spec['hidden_full_bias_%i' % (i + 1)] = outlayer
        if rec:
            size_hidden = image_shape[-1]
            spec = define_recurrent_params(spec, i+len(n_hidden_conv), size_hidden, rec)
            spec['initial_hiddens_%i' % (i+len(n_hidden_conv))] = size_hidden

    print spec
    return spec


def recurrent_layer(hidden_inpt, hidden_to_hidden, f, initial_hidden,
                    rec_shape, rec_type):
    def step_full(x, hi_tm1):
        h_tm1 = f(hi_tm1)
        hi = T.dot(h_tm1, hidden_to_hidden) + x
        return hi
    def step_single(x, hi_tm1):
        h_tm1 = f(hi_tm1)
        hi = (h_tm1 * hidden_to_hidden) + x
        return hi
    def step_double(x, hi_tm1):
        h_tm1 = f(hi_tm1)
        hi = T.dot(T.dot(h_tm1, hidden_to_hidden[0]), hidden_to_hidden[1]) + x
        return hi
    def step_conv(x, hi_tm1):
        h_tm1 = f(hi_tm1)
        h_tm1 = h_tm1.reshape((rec_shape[1], 1, rec_shape[2], 1))
        h_tm1 = conv.conv2d(h_tm1, hidden_to_hidden, filter_shape=(1, 1, rec_type[1], 1),
                            image_shape=(rec_shape[1], 1, rec_shape[2], 1), border_mode='full')
        h_tm1 = h_tm1[:, :, :rec_shape[2], :]
        h_tm1 = h_tm1.reshape((rec_shape[1], rec_shape[2]))
        hi = h_tm1 + x
        return hi

    if rec_type == 'full':
        step = step_full
    elif rec_type == 'single':
        step = step_single
    elif rec_type[0] == 'double':
        step = step_double
    elif rec_type[0] == 'conv':
        step = step_conv
    else:
        raise AttributeError('Recurrency format is not correct')
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

    return hidden_in_rec


def conv_part(inpt, params, img_shape):
    w, b, fs, ps = params
    hidden_in = conv.conv2d(inpt, w, filter_shape=fs,
                            image_shape=img_shape)
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
    output_flat = T.dot(inpt_flat, weights)
    output = output_flat.reshape((n_time_steps, n_samples, n_output))
    output += bias.dimshuffle('x', 'x', 0)
    return output


def exprs(inpt, target, in_to_hidden, hidden_to_out, out_bias,
          hidden_conv_to_hidden_full, hidden_conv_to_hidden_conv,
          hidden_full_to_hidden_full, hidden_conv_bias,
          hidden_full_bias, hidden_conv_transfers,
          hidden_full_transfers, output_transfer, loss,
          image_shapes, filter_shapes_comp,
          pool_shapes, recurrents, initial_hiddens, weights,
          recurrent_types):

    print image_shapes
    #input shape = n_time_steps, n_samples, channels, n_frames_to_take, n_output
    #conv part: reshape to n_time_step * n_samples, channels, n_frames_to_take, n_output
    #rec part: reshape to n_time_steps, n_samples, channels * n_frames_to_take * n_output
    exprs = {}

    # Convolutional part
    zipped = zip(image_shapes[1:], hidden_conv_transfers,
                 recurrents, recurrent_types, initial_hiddens,
                 [in_to_hidden] + hidden_conv_to_hidden_conv, hidden_conv_bias,
                 filter_shapes_comp, pool_shapes)

    conv_shape = [np.prod(image_shapes[0][:2])] + list(image_shapes[0][2:])
    hidden = inpt.reshape(conv_shape)
    for i, params in enumerate(zipped):
        image_shape, ft, rec, rec_type, ih = params[:5]
        f = lookup(ft, transfer)
        hidden_in_down = conv_part(hidden, params[5:], conv_shape)
        conv_shape = [np.prod(image_shape[:2])] + list(image_shape[2:])
        if rec is not None:
            rec_shape = list(image_shape[:2]) + [np.prod(image_shape[2:])]
            reshaped_hidden_in_conv_down = (hidden_in_down.reshape(image_shape)).reshape(rec_shape)
            hidden_in_rec = recurrent_layer(reshaped_hidden_in_conv_down, rec, f, ih, rec_shape, rec_type)
            hidden_in_down = (hidden_in_rec.reshape(image_shape)).reshape(conv_shape)
        exprs['conv-hidden_in_%i' % i] = hidden_in_down
        hidden = exprs['conv-hidden_%i' % i] = f(hidden_in_down)

    # Mlp part
    offset = len(hidden_conv_bias)
    zipped = zip([hidden_conv_to_hidden_full] + hidden_full_to_hidden_full,
                 hidden_full_bias, hidden_full_transfers, recurrents[offset:],
                 recurrent_types[offset:], initial_hiddens[offset:],
                 image_shapes[offset+1:])
    image_shape = image_shapes[offset]
    rec_shape = list(image_shape[:2]) + [np.prod(image_shape[2:])]
    hidden = hidden.reshape(rec_shape)
    for i, (w, b, t, rec, rec_type, ih, image_shape) in enumerate(zipped):
        hidden_in = feedforward_layer(hidden, w, b)
        f = lookup(t, transfer)
        if rec is not None:
            hidden_in = recurrent_layer(hidden_in, rec, f, ih, image_shape, rec_type)
        hidden = f(hidden_in)
        exprs['hidden_in_%i' % (i + offset + 1)] = hidden_in
        exprs['hidden_%i' % (i + offset + 1)] = hidden

    f_output = lookup(output_transfer, transfer)

    output_in = feedforward_layer(hidden, hidden_to_out, out_bias)

    output = f_output(output_in)

    f_loss = lookup(loss, loss_)
    loss_coordwise = f_loss(target, output)


    if weights is not None:
        loss_coordwise *= weights
    loss_samplewise = loss_coordwise.sum(axis=2)
    if weights is not None:
        weights_samplewise = weights.mean(axis=2)
        overall_loss = loss_samplewise.sum(axis=None) / weights_samplewise.sum(axis=None)
    else:
        overall_loss = loss_samplewise.mean()

    exprs.update({
        'loss_samplewise': loss_samplewise,
        'loss': overall_loss,
        'inpt': inpt,
        'target': target,
        'output_in': output_in,
        'output': output
    })

    return exprs
