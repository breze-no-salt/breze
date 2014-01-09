
# -*- coding: utf-8 -*-


import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample


from ...util import lookup
from ...component import transfer, loss as loss_


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
        exprs['hidden_in_%i' % (i + 1)] = hidden_in_down
        exprs['hidden_%i' % (i + 1)] = hidden

    # Mlp part
    f_hidden = lookup(hidden_full_transfers[0], transfer)
    hidden_middle = hidden.flatten(2)
    offset = len(hidden_conv_to_hidden_conv)
    hidden_in = T.dot(hidden_middle, hidden_conv_to_hidden_full) + hidden_full_bias[0]
    exprs['hidden_in_0'] = hidden_in
    hidden = exprs['hidden_0'] = f_hidden(hidden_in)
    zipped = zip(hidden_full_to_hidden_full, hidden_full_bias[1:], hidden_full_transfers[1:])
    for i, (w, b, t) in enumerate(zipped):
        hidden_m1 = hidden
        hidden_in = T.dot(hidden_m1, w) + b
        f = lookup(t, transfer)
        hidden = f(hidden_in)
        exprs['hidden_in_%i' % (i + offset + 1)] = hidden_in
        exprs['hidden_%i' % (i + offset + 1)] = hidden

    f_output = lookup(output_transfer, transfer)
    output_in = T.dot(hidden, hidden_to_out) + out_bias
    output = f_output(output_in)

    f_loss = lookup(loss, loss_)

    loss_rowwise = f_loss(target, output).sum(axis=1)
    loss = loss_rowwise.mean()

    exprs.update({
        'inpt': inpt,
        'target': target,
        'output_in': output_in,
        'output': output,
        'loss_rowwise': loss_rowwise,
        'loss': loss
    })

    return exprs
