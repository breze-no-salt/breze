# -*- coding: utf-8 -*-

import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from ...util import lookup
from ...component import transfer, loss as loss_

import mlp


def pad(inpt_to_pad, pad_to_add):
    if pad_to_add == 0:
        return inpt_to_pad
    dim2 = T.zeros_like(inpt_to_pad[:, :, :pad_to_add, :])
    padded_output = T.concatenate([dim2, inpt_to_pad, dim2], axis=2)
    dim3 = T.zeros_like(padded_output[:, :, :, :pad_to_add])
    return T.concatenate([dim3, padded_output, dim3], axis=3)


def perform_pooling(tensor, shift, pool_shape, limits):
    if pool_shape[0] == 1 and pool_shape[1] == 1:
        return tensor
    hidden_in_conv_predown = T.zeros_like(tensor[:, :, :limits[0], :limits[1]])
    shift_i, shift_j = shift
    skip_i = (shift_i[-1] / pool_shape[0]) + 1
    skip_j = (shift_j[-1] / pool_shape[1]) + 1
    for idi, i in enumerate(shift_i):
        for idj, j in enumerate(shift_j):
            partial_downsampled = downsample.max_pool_2d(
                tensor[:, :, i:, j:], pool_shape,
                ignore_border=True
            )
            hidden_in_conv_predown = T.set_subtensor(
                hidden_in_conv_predown[:, :, idi::len(shift_i), idj::len(
                    shift_j)],
                partial_downsampled[:, :, ::skip_i, ::skip_j]
            )
    return hidden_in_conv_predown


def perform_lrnorm(inpt, lrnorm):
    alpha, beta, N = lrnorm
    limit = N / 2
    squared_inpt = T.sqr(inpt)
    final_result = squared_inpt.copy()
    for i in range(limit + 1):
        for j in range(limit + 1):
            if i == 0 and j == 0:
                continue
            elif i == 0:
                final_result = T.inc_subtensor(final_result[:, :, :, :-j],
                                               squared_inpt[:, :, :, j:])
                final_result = T.inc_subtensor(final_result[:, :, :, j:],
                                               squared_inpt[:, :, :, :-j])
            elif j == 0:
                final_result = T.inc_subtensor(final_result[:, :, :-i, :],
                                               squared_inpt[:, :, i:, :])
                final_result = T.inc_subtensor(final_result[:, :, i:, :],
                                               squared_inpt[:, :, :-i, :])
            else:
                final_result = T.inc_subtensor(final_result[:, :, :-i, :-j],
                                               squared_inpt[:, :, i:, j:])
                final_result = T.inc_subtensor(final_result[:, :, i:, j:],
                                               squared_inpt[:, :, :-i, :-j])
                final_result = T.inc_subtensor(final_result[:, :, :-i, j:],
                                               squared_inpt[:, :, i:, :-j])
                final_result = T.inc_subtensor(final_result[:, :, i:, :-j],
                                               squared_inpt[:, :, :-i, j:])
    final_result *= T.constant((alpha + 0.0) / (N * N))
    final_result += 1
    return inpt / T.pow(final_result, beta)


def convolution_part(inpt, padding, image_shapes, weights, filter_shape,
                     pool_shape, pool_shift, bias, transfer_name, lrnom):
    padded_inpt = pad(inpt, padding[0])
    f_hidden = lookup(transfer_name, transfer)
    in_conv = conv.conv2d(padded_inpt, weights, filter_shape=filter_shape,
                          image_shape=image_shapes[0])
    shape_before_pooling = [(image_shapes[1][-2] - 2 * padding[1]),
                            (image_shapes[1][-1] - 2 * padding[1])]
    conv_predown = perform_pooling(in_conv, pool_shift, pool_shape,
                                   shape_before_pooling)
    conv_down = (conv_predown + bias.dimshuffle('x', 0, 'x', 'x'))
    if lrnom is not None:
        conv_down = perform_lrnorm(conv_down, lrnom)
    return conv_down, f_hidden(conv_down)


def parameters(n_inpt, n_hidden_conv, n_hidden_full, n_output,
               resulting_image_size, filter_shapes):
    spec = dict(in_to_hidden=(n_hidden_conv[0], n_inpt[1],
                              filter_shapes[0][0], filter_shapes[0][1]),
                hidden_to_out=(n_hidden_full[-1], n_output),
                hidden_conv_to_hidden_full=(n_hidden_conv[-1] *
                                            resulting_image_size,
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
          image_shapes, filter_shapes_comp, input_shape,
          pool_shapes, pool_shift, padding, lrnorm):
    exprs = {}

    reshaped_inpt = inpt.reshape(input_shape)
    conv_part = convolution_part(
        reshaped_inpt, padding[:2], image_shapes[:2], in_to_hidden,
        filter_shapes_comp[0], pool_shapes[0], pool_shift[0],
        hidden_conv_bias[0], hidden_conv_transfers[0], lrnorm[0]
    )
    exprs['hidden_in_0'], exprs['hidden_0'] = conv_part
    hidden = exprs['hidden_0']

    zipped = zip(hidden_conv_to_hidden_conv, hidden_conv_bias[1:],
                 hidden_conv_transfers[1:], filter_shapes_comp[1:],
                 pool_shapes[1:], pool_shift[1:])
    for i, (w, b, t, fs, psh, psf) in enumerate(zipped):
        conv_part = convolution_part(
            hidden, padding[i + 1:i + 3], image_shapes[i + 1:i + 3],
            w, fs, psh, psf, b, t, lrnorm[i + 1]
        )
        exprs['conv-hidden_in_%i' % (i + 1)] = conv_part[0]
        exprs['conv-hidden_%i' % (i + 1)] = conv_part[1]
        hidden = conv_part[1]

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