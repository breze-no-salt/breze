# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet.bn import batch_normalization
from theano.tensor.signal import downsample
from theano.ifelse import ifelse

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


class Conv2d(Layer):

    def __init__(self, inpt, inpt_height, inpt_width, n_inpt,
                 filter_height, filter_width,
                 n_output, transfer='identity',
                 n_samples=None,
                 subsample=(1, 1),
                 declare=None, name=None):
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.n_inpt = n_inpt

        self.filter_height = filter_height
        self.filter_width = filter_width

        self.n_output = n_output
        self.transfer = transfer
        self.n_samples = n_samples
        self.subsample = subsample

        # self.output_height, _ = divmod(inpt_height, filter_height)
        # self.output_width, _ = divmod(inpt_width, filter_width)
        self.output_height = (inpt_height - filter_height) / subsample[0] + 1
        self.output_width = (inpt_width - filter_width) / subsample[1] + 1

        if not self.output_height > 0:
            raise ValueError('inpt height smaller than filter height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than filter width')

        super(Conv2d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.weights = self.declare((
            self.n_output, self.n_inpt,
            self.filter_height, self.filter_width))
        self.bias = self.declare((self.n_output,))

        self.output_in = conv.conv2d(
            self.inpt, self.weights,
            image_shape=(
                self.n_samples,
                self.n_inpt,
                self.inpt_height,
                self.inpt_width
            ),
            subsample=self.subsample,
            border_mode='valid',
            )

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)


class MaxPool2d(Layer):

    def __init__(self, inpt, inpt_height, inpt_width, pool_height, pool_width,
                 n_output,
                 transfer='identity',
                 declare=None, name=None):
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.transfer = transfer

        self.output_height, _ = divmod(inpt_height, pool_height)
        self.output_width, _ = divmod(inpt_width, pool_width)

        if not self.output_height > 0:
            raise ValueError('inpt height smaller than pool height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than pool width')

        self.n_output = n_output

        super(MaxPool2d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.output_in = downsample.max_pool_2d(
            input=self.inpt, ds=(self.pool_height, self.pool_width),
            ignore_border=True)

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)


class BatchNormalization(Layer):
    """Class implementing Batch Normalization (BN) [D] adapted to fully
    connected layers.

    If X is the input, computes:
        BN(X) = scale * (X - mean) / \sqrt(std + num_stability_cst) + shift
    where scale and shift are learnable parameters, and:
        * at training time, mean and std are computed on the mini-batch
        samples, while keeping track of the exponential moving average
        of mean and std.
        * at validation time, the exponential moving average of mean and
        std are used.

    To apply Batch Normalization to a layer L, one should:
        1. set the transfer function of the layer L to "identity" and the
        transfer function of BN layer to the one of the layer L.
        2. remove the bias of the layer L.

    For image layers (convolutional layers, maxpool layers, etc.), please use
    BatchNormalization2d.

    References
    ----------
    .. [D] Ioffe, S., & Szegedy, C. (2015).
           Batch normalization: Accelerating deep network training by
           reducing internal covariate shift.
           arXiv preprint arXiv:1502.03167.

    Attributes
    ----------
    training : int
        whether the network is in training phase
        set to 0 if not training
    weighting_decrease : float
        degree of weighting decrease in the exponential moving average
        computation of mean and std, must be between 0 and 1
    num_stability_cst : float
        constant used for numerical stability, called epsilon in [D].
    scale : vector of floats
        factor to scale the normalized input, called gamma in [D].
        learnable parameter
    shift : vector of floats
        factor to shift the normalized input, called beta in [D].
        learnable parameter
    """

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, training):
        self._training = training

    def __init__(self, inpt, n_inpt,
                 n_samples,
                 weighting_decrease=0.3,
                 training=1,
                 transfer='identity',
                 declare=None, name=None):

        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_output = n_inpt
        self.n_samples = n_samples
        self.transfer = transfer

        if weighting_decrease < 0 or weighting_decrease > 1:
            raise ValueError("weighting_decrease must be between O and 1")

        self.weighting_decrease = weighting_decrease

        self._training = training

        self.num_stability_cst = 1e-6

        super(BatchNormalization, self).__init__(declare=declare, name=name)

    def _forward(self):

        self.scale = self.declare((self.n_inpt,))
        self.shift = self.declare((self.n_inpt,))

        self.mean = theano.shared(
            np.zeros((self.n_inpt,), dtype=theano.config.floatX),
            "mean"
        )
        self.std = theano.shared(
            np.ones((self.n_inpt,), dtype=theano.config.floatX),
            "std"
        )

        mean = ifelse(
            self.training,
            self.inpt.mean(axis=0),
            self.mean
        )

        std = ifelse(
            self.training,
            self.inpt.std(axis=0) + self.num_stability_cst,
            self.std
        )

        self.mean.default_update = ifelse(
            self.training,
            (self.weighting_decrease*mean
             + (1 - self.weighting_decrease)*self.mean),
            self.mean
        )

        self.std.default_update = ifelse(
            self.training,
            (self.weighting_decrease*std
             + (1 - self.weighting_decrease)*self.std),
            self.std
        )

        self.output_in = batch_normalization(
            self.inpt,
            self.scale,
            self.shift,
            mean,
            std,
            "low_mem"
        )

        f = lookup(self.transfer, _transfer)

        self.output = f(self.output_in)


class BatchNormalization2d(Layer):
    """Class implementing Batch Normalization (BN) [D] adapted to image layers.

    If X is the input, computes:
        BN(X) = scale * (X - mean) / \sqrt(std + num_stability_cst) + shift
    where scale and shift are learnable parameters, and:
        * at training time, mean and std are computed on the mini-batch
        samples, while keeping track of the exponential moving average
        of mean and std.
        * at validation time, the exponential moving average of mean and
        std are used.

    To apply Batch Normalization to a layer L, one should:
        1. set the transfer function of the layer L to "identity" and the
        transfer function of BN layer to the one of the layer L.
        2. remove the bias of the layer L.

    For fully connected layers, please use BatchNormalization.

    References
    ----------
    .. [D] Ioffe, S., & Szegedy, C. (2015).
           Batch normalization: Accelerating deep network training by
           reducing internal covariate shift.
           arXiv preprint arXiv:1502.03167.

    Attributes
    ----------
    training : int
        whether the network is in training phase
        set to 0 if not training
    weighting_decrease : float
        degree of weighting decrease in the exponential moving average
        computation of mean and std, must be between 0 and 1
    num_stability_cst : float
        constant used for numerical stability, called epsilon in [D].
    scale : vector of floats
        factor to scale the normalized input, called gamma in [D].
        learnable parameter
    shift : vector of floats
        factor to shift the normalized input, called beta in [D].
        learnable parameter
    """
    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, training):
        self._training = training

    def __init__(self, inpt, inpt_height, inpt_width,
                 n_output, n_samples,
                 weighting_decrease=0.3,
                 training=1,
                 transfer='identity',
                 declare=None, name=None):

        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.output_height = inpt_height
        self.output_width = inpt_width
        self.n_output = n_output
        self.n_samples = n_samples
        self.transfer = transfer

        if weighting_decrease < 0 or weighting_decrease > 1:
            raise ValueError("weighting_decrease must be between O and 1")

        self.weighting_decrease = weighting_decrease

        self._training = training

        self.num_stability_cst = 1e-6

        super(BatchNormalization2d, self).__init__(declare=declare, name=name)

    def _forward(self):

        self.scale = self.declare((1, self.n_output, 1, 1))
        self.shift = self.declare((1, self.n_output, 1, 1))

        self.mean = theano.shared(
            np.zeros((1, self.n_output, 1, 1), dtype=theano.config.floatX),
            "mean"
        )
        self.std = theano.shared(
            np.ones((1, self.n_output, 1, 1), dtype=theano.config.floatX),
            "std"
        )

        axes = (0, 2, 3)

        mean = ifelse(
            self.training,
            # unbroadcast is necessary otherwise it would not be
            # of the same type as self.mean
            T.unbroadcast(
                self.inpt.mean(axis=axes, keepdims=True),
                *(0, 1, 2, 3)
            ),
            self.mean
        )

        std = ifelse(
            self.training,
            # unbroadcast is necessary otherwise it would not be
            # of the same type as self.std
            T.unbroadcast(
                self.inpt.std(axis=axes, keepdims=True),
                *(0, 1, 2, 3)
            ) + self.num_stability_cst,
            self.std
        )

        self.mean.default_update = ifelse(
            self.training,
            (self.weighting_decrease*mean
             + (1 - self.weighting_decrease)*self.mean),
            self.mean
        )

        self.std.default_update = ifelse(
            self.training,
            (self.weighting_decrease*std
             + (1 - self.weighting_decrease)*self.std),
            self.std
        )

        # the axes should be broadcastable for computation
        mean = T.addbroadcast(mean, *axes)
        std = T.addbroadcast(std, *axes)

        self.output_in = batch_normalization(
            self.inpt,
            self.scale,
            self.shift,
            mean,
            std,
            "low_mem"
        )

        f = lookup(self.transfer, _transfer)

        self.output = f(self.output_in)
