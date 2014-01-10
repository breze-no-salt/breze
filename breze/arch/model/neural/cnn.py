
# -*- coding: utf-8 -*-


import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from ...util import lookup
from ...component import transfer, loss as loss_

import mlp


def parameters(n_inpt, n_hidden_conv, n_hidden_full, n_output,
               resulting_image_size, filter_shapes):
    spec = dict(in_to_hidden=(n_hidden_conv[0], n_inpt[1],
                filter_shapes[0][0], filter_shapes[0][1]),
                hidden_to_out=(n_hidden_full[-1], n_output),
                hidden_conv_to_hidden_full=(n_hidden_conv[-1] * resulting_image_size,
                                            n_hidden_full[0]),
                hidden_conv_bias_0=n_hidden_conv[0],
                hidden_full_bias_0=n_hidden_full[0],
                out_bias=n_output)
    zipped = zip(n_hidden_conv[:-1], n_hidden_conv[1:], filter_shapes[1:])
    for i, (inlayer, outlayer, filter_shape) in enumerate(zipped):
        spec['hidden_conv_to_hidden_conv_%i' % i] = (
            outlayer, inlayer, filter_shape[0], filter_shape[1])
        spec['hidden_conv_bias_%i' % (i + 1)] = outlayer
    zipped = zip(n_hidden_full[:-1], n_hidden_full[1:])
    for i, (inlayer, outlayer) in enumerate(zipped):
        spec['hidden_full_to_hidden_full_%i' % i] = (inlayer, outlayer)
        spec['hidden_full_bias_%i' % (i + 1)] = outlayer
    return spec


def exprs(inpt, target, in_to_hidden, hidden_to_out, out_bias,
          hidden_conv_to_hidden_full, hidden_conv_to_hidden_conv,
          hidden_full_to_hidden_full, hidden_conv_bias,
          hidden_full_bias, hidden_conv_transfers,
          hidden_full_transfers, output_transfer, loss,
          image_shapes, filter_shapes_comp, input_shape, pool_size):
    exprs = {}

    reshaped_inpt = inpt.reshape(input_shape)
    f_hidden = lookup(hidden_conv_transfers[0], transfer)

    # Convolutional part
    hidden_in_conv = conv.conv2d(
        reshaped_inpt, in_to_hidden, filter_shape=filter_shapes_comp[0],
        image_shape=image_shapes[0])
    hidden_in_conv_predown = downsample.max_pool_2d(
        hidden_in_conv, pool_size, ignore_border=True)
    hidden_in_conv_down = (hidden_in_conv_predown
                           + hidden_conv_bias[0].dimshuffle('x', 0, 'x', 'x'))
    exprs['hidden_in_0'] = hidden_in_conv_down
    hidden = exprs['hidden_0'] = f_hidden(hidden_in_conv_down)

    zipped = zip(hidden_conv_to_hidden_conv, hidden_conv_bias[1:],
                 hidden_conv_transfers[1:], filter_shapes_comp[1:],
                 image_shapes[1:])
    for i, (w, b, t, fs, ims) in enumerate(zipped):
        hidden_m1 = hidden
        hidden_in = conv.conv2d(hidden_m1, w, filter_shape=fs,
                                image_shape=ims)
        hidden_in_predown = downsample.max_pool_2d(
            hidden_in, pool_size, ignore_border=True)
        hidden_in_down = hidden_in_predown + b.dimshuffle('x', 0, 'x', 'x')
        f = lookup(t, transfer)
        hidden = f(hidden_in_down)
        exprs['conv-hidden_in_%i' % (i + 1)] = hidden_in_down
        exprs['conv-hidden_%i' % (i + 1)] = hidden

    # Mlp part
    hidden_middle = hidden.flatten(2)

    exprs.update(mlp.exprs(
        hidden_middle, hidden_conv_to_hidden_full,
        hidden_full_to_hidden_full,
        hidden_to_out, hidden_full_bias, out_bias,
        hidden_full_transfers, output_transfer,
        prefix='mlp-'))

    f_loss = lookup(loss, loss_)

    # Tidy a little.
    exprs['output'] = exprs['mlp-output']
    exprs['output'].name = 'output'
    del exprs['mlp-output']
    exprs['output_in'] = exprs['mlp-output_in']
    del exprs['mlp-output_in']
    exprs['output_in'].name = 'output_in'

    loss_rowwise = f_loss(target, exprs['output']).sum(axis=1)
    loss = loss_rowwise.mean()

    exprs.update({
        'loss_rowwise': loss_rowwise,
        'loss': loss
    })

    return exprs
