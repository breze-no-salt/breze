# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T
import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from breze.arch.component import transfer as _transfer, loss as _loss
from breze.arch.construct.base import Layer
from breze.arch.util import lookup


class AffineNonlinear(Layer):

    @property
    def n_inpt(self):
        return self._n_inpt

    @property
    def n_output(self):
        return self._n_output

    def __init__(self, inpt, n_inpt, n_output, transfer='identity',
                 use_bias=True, declare=None, name=None):
        self.inpt = inpt
        self._n_inpt = n_inpt
        self._n_output = n_output
        self.transfer = transfer
        self.use_bias = use_bias
        super(AffineNonlinear, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.weights = self.declare((self.n_inpt, self.n_output))

        self.output_in = T.dot(self.inpt, self.weights)

        if self.use_bias:
            self.bias = self.declare(self.n_output)
            self.output_in += self.bias

        f = lookup(self.transfer, _transfer)

        self.output = f(self.output_in)


class Split(Layer):

    def __init__(self, inpt, lengths, axis=1, declare=None, name=None):
        self.inpt = inpt
        self.lengths = lengths
        self.axis = axis
        super(Split, self).__init__(declare, name)

    def _forward(self):
        starts = [0] + np.add.accumulate(self.lengths).tolist()
        stops = starts[1:]
        starts = starts[:-1]

        self.outputs = [self.inpt[:, start:stop] for start, stop
                        in zip(starts, stops)]


class Concatenate(Layer):

    def __init__(self, inpts, axis=1, declare=None, name=None):
        self.inpts = inpts
        self.axis = axis
        super(Concatenate, self).__init__(declare, name)

    def _forward(self):
        concatenated = T.concatenate(self.inpts, self.axis)
        self.output = concatenated


class SupervisedLoss(Layer):

    def __init__(self, target, prediction, loss, comp_dim=1, imp_weight=None,
                 declare=None, name=None):
        self.target = target
        self.prediction = prediction
        self.loss_ident = loss

        self.imp_weight = imp_weight
        self.comp_dim = comp_dim

        super(SupervisedLoss, self).__init__(declare, name)

    def _forward(self):
        f_loss = lookup(self.loss_ident, _loss)

        self.coord_wise = f_loss(self.target, self.prediction)

        if self.imp_weight is not None:
            self.coord_wise *= self.imp_weight

        self.sample_wise = self.coord_wise.sum(self.comp_dim)

        self.total = self.sample_wise.mean()


class Convolution(Layer):
"""
implements one convolutional plus pooling layer
"""
    def __init__(self, inpt, image_shape_in, image_shape_out, filter_shape, pool_shape, pool_shift, transfer, padding, lrnorm,
                 use_bias=True, declare=None, name=None):

        assert image_shape_in[1] == filter_shape[1]
        self.inpt = inpt
        self.image_shape_in = image_shape_in
        self.image_shape_out = image_shape_out
        self.filter_shape = filter_shape
        self.pool_shape = pool_shape
        self.pool_shift = pool_shift
        self.transfer = transfer
        self.padding = padding
        self.lrnorm = lrnorm
        self.use_bias = use_bias
        super(Convolution, self).__init__(declare=declare, name=name)


    def _forward(self):

        self.weights = self.declare(self.filter_shape)

        padded_inpt = pad(self.inpt, self.padding[0])
        f_hidden = lookup(self.transfer, _transfer)
        in_conv = conv.conv2d(padded_inpt, self.weights, filter_shape=self.filter_shape,
                              image_shape=self.image_shape_in)
        shape_before_pooling = [(self.image_shape_out[-2] - 2 * self.padding[1]),
                                (self.image_shape_out[-1] - 2 * self.padding[1])]
        conv_predown = perform_pooling(in_conv, self.pool_shift, self.pool_shape,
                                       shape_before_pooling)
        if self.use_bias:
            self.bias = self.declare((self.filter_shape[0],))
            # add the bias term. Since the bias is a vector (1D array), we first
            # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
            # thus be broadcasted across mini-batches and feature map
            # width & height
            conv_down = (conv_predown + self.bias.dimshuffle('x', 0, 'x', 'x'))

        if self.lrnorm is not None:
            conv_down = perform_lrnorm(conv_down, self.lrnom)

        self.output = f_hidden(conv_down)


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