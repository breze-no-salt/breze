# -*- coding: utf-8 -*-

import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from ...util import lookup
from ...component import transfer, loss as loss_

import mlp


def pad(inpt_to_pad, pad_to_add):
    """Zero-pad a the last two dimensions of a 4-dimensional Theano variable.

    Parameters
    ----------

    inpt_to_pad : Theano variable
        Variable to which the pad is added.

    pad_to_add : Integer
        Number of zeros to pad in every direction.

    Returns
    -------

    res : Theano variable
        Padded variable by the specified padding.
    """
    if pad_to_add == 0:
        return inpt_to_pad
    dim2 = T.zeros_like(inpt_to_pad[:, :, :pad_to_add, :])
    padded_output = T.concatenate([dim2, inpt_to_pad, dim2], axis=2)
    dim3 = T.zeros_like(padded_output[:, :, :, :pad_to_add])
    return T.concatenate([dim3, padded_output, dim3], axis=3)


def perform_pooling(tensor, shift, pool_shape, limits):
    """Perform max-pooling over the last two dimensions of a 4-dimensional
    Theano variable. In order to pool with overlapping patches, the pooling
    can be performed over interleaved shifts of the image as desired.

    Parameters
    ----------

    tensor : Theano variable
        Variable on which the last two dimensions are max-pooled.

    shift : List of lists of integers
        One list per dimension with the image shifts in that direction.
        E.g. [[0],[0]] for non-overlapping pooling, [[0, 1], [0]] for
        separation of 1 between patches in the 3rd dimension,
        and non-overlapping in the 4th dimension.

    pool_shape : List of integers
        Patch size of the pooling.

    limits : List of integers
        Size of the variable after pooling.

    Returns
    -------

    res : Theano variable
        Input variable after max-pooling has been applied.

    """
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


#TODO: implement this with a convolution
def perform_lrnorm(inpt, lrnorm):
    """Perform local response normalization in the same map

    Parameters
    ----------

    inpt : Theano variable
        Variable on which local response normalization is applied (last two
        dimensions).

    lrnom : List of floats
        Parameters of the local response normalization (alpha, beta, size of
        patch).

    Returns
    -------

    res : Theano variable
        Input variable after local response normalization has been applied.

    """
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
    """Applies a convolutional layer: convolution, pooling (optional) and
    local response normalization (optional).

    Parameters
    ----------

    inpt : Theano variable
        Variable representing the input to this part.

    padding : List of integers
        Pad to add to the input of this annd the following layer.

    image_shapes : List of integers
        List of the calculated shapes of the inputs of every layer.

    weights : Theano variable
        Parameters that are convolved with the input in the convolutional
        layer.

    filter_shape : List of integers
        Shape of the filters (i.e. kernels) that are applied to the input in
        the convolutional layer.

    pool_shape : List of integers
        Shape of the patches in the pooling layer.

    pool_shift : List of lists of integers
        Each item represents a shift of the image by a certain offset,
        such that overlapped pooling is achieved in the pooling layer.

    bias : Theano variable.
        Bias to be added in this layer.

    transfer_name : String
        Name of the transfer function applied in this layer.

    lrnorm : List of floats.
        List of parameters for local response normalization. If None,
        local response normalization is not applied.

    prefix : string, optional [default: '']
        The key of each expression will be prefixed with this string in the
        result dict.


    Returns
    -------

    exprs : Tuple with the output variable of that layer, before and after
    applying the transfer function.

    """
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
    """Return the parameter specification dictionary for a CNN.

    Parameters
    ----------

    n_inpt : Integer
        Number of inputs of the model.

    n_hidden_conv : List of integers
        Each item corresponds to the number of feature maps of a
        convolutional layer.

    n_hidden_full : List of integers
        Each item corresponds to the number of neurons of a fully connected
        layer.

    n_output : Integer
        Number of outputs of the model.

    resulting_image_size : Integer
        Computed size of the last convolutional layer in neurons. Used in
        order to ease transition to fully connected layers.

    filter_shapes : List of integers
        List of the shapes of the filters to be applied in the convolutional
        layers.

    Returns
    -------

    res : dict
        Dictionary specifying the parameters needed.
    """
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
          pool_shapes, pool_shifts, padding, lrnorm):
    """Return the expressions for a CNN.

    Parameters
    ----------

    inpt : Theano variable
        Variable representing the input to the model.

    target : Theano variable
        Variable that represents the target values with respect to which the
        loss is calculated.

    in_to_hidden : Theano variable
        Variable represeting the input to the first convolutional layer weight
        matrix.

    hidden_to_out : Theano variable
        Variable representing the last layer to output weight matrix.

    out_bias : Theano variable
        Output bias.

    hidden_conv_to_hidden_full : Theano variable
        Variable representing the convolutional layer to fully connected
        layer weight matrix.

    hidden_conv_to_hidden_conv : List of Theano variables
        Each variable in the list represents the weights between two
        convolutional layers (in order, closest layer to input first).

    hidden_full_to_hidden_full : List of Theano variables
        Each variable in the list represents the weights between two
        fully connected layers (in order, closest layer to input first).

    hidden_conv_bias : List of Theano variables
        Each variable in the list represents the biases to add in every
        convolutional layer.

    hidden_full_bias : List of Theano variables
        Each variable in the list represents the biases to add in every
        fully connected layer.

    hidden_conv_transfers : List of strings or callables.
        Each item is a transfer function mapping a Theano matrix to a Theano
        matrix of the same size in the convolutional layers. If a string,
        such a function will be looked up by that name in
        ``breze.arch.component.transfer``.

    hidden_full_transfers : List of strings or callables.
        Each item is a transfer function mapping a Theano matrix to a Theano
        matrix of the same size in the fully connected layers. If a string,
        such a function will be looked up by that name in  ``breze.arch
        .component.transfer``.

    output_transfer : String or callable.
        Output transfer function mapping a Theano matrix to a Theano matrix of
        the same size. If a string, such a function will be looked up by that
        name in ``breze.arch.component.transfer``.

    loss : String or callable
        Loss to calculate between the output of the model and the specified
        target.

    image_shapes : List of integers
        Pre-calculated size of the input at for every convolutional layers.
        Convolution operator in Theano needs it in compile time.

    filter_shapes_comp : List of integers
        Pre-calculated size of the filters (kernels) applied to the input
        of every convolutional layer. Convolution operator in Theano needs it
        in compile time.

    input_shape : List of integers
        Shape of the input in 4-dimensional structure, used to reshape the
        input.

    pool_shapes : List of integers
        Sizes of the patches in the pooling layers.

    pool_shifts : List of integers
        Shifts of the input to be used in pooling layers in order to achieve
        overlapped pooling.

    padding : List of integers
        Each element of the list represents the pad to add to the
        corresponding convolutional layer.

    lrnorm : List of floats
        Group of parameters of the local response normalization
        applied to each layer.

    Returns
    -------

    exprs : Dictionary with expressions of the model.
    """
    exprs = {}

    reshaped_inpt = inpt.reshape(input_shape)
    conv_part = convolution_part(
        reshaped_inpt, padding[:2], image_shapes[:2], in_to_hidden,
        filter_shapes_comp[0], pool_shapes[0], pool_shifts[0],
        hidden_conv_bias[0], hidden_conv_transfers[0], lrnorm[0]
    )
    exprs['hidden_in_0'], exprs['hidden_0'] = conv_part
    hidden = exprs['hidden_0']

    zipped = zip(hidden_conv_to_hidden_conv, hidden_conv_bias[1:],
                 hidden_conv_transfers[1:], filter_shapes_comp[1:],
                 pool_shapes[1:], pool_shifts[1:])
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