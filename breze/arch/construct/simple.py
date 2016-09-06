# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

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
                self.n_samples, self.n_inpt,
                self.inpt_height, self.inpt_width
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


class Dropout(Layer):
    """Class representing a Dropout layer [D] (section 3.3).

    At training time, a unit is kept with probability p.
    At test time, the weights are multiplied by p, giving the
    same output as the expected output at training time.

    References
    ----------
    .. [D] Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I.,
                   & Salakhutdinov, R. R. (2012).
                   Improving neural networks by preventing co-adaptation
                   of feature detectors.
                   arXiv preprint arXiv:1207.0580.

    Attributes
    ----------
    training : int
        whether the network is in training phase
        set to 0 if not training

    rate : float
        probability of not dropping out a unit

    """

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, training):
        self._training = training
        
    def __init__(self, inpt, 
                 n_output,
                 rng,
                 training,
                 rate,
                 inpt_height = None,
                 inpt_width = None,
                 transfer='identity',
                 declare=None, name=None):

        self.inpt = inpt

        self.output_height = inpt_height
        self.output_width = inpt_width

        self.transfer = transfer

        self.n_output = n_output

        self.srng = RandomStreams(rng.randint(2**32))
        self._training = training

        self.rate = rate

        super(Dropout, self).__init__(declare=declare, name=name)

    def _forward(self):

        mask = self.srng.binomial(
            n=1, p=self.rate, size=self.inpt.shape,
            dtype=theano.config.floatX
        )

        # if training is different than 0, then we drop out units
        # else we multiply the weights by rate

        self.output_in = ifelse(
            self.training,
            self.inpt * mask,
            self.inpt * self.rate
        )

        f = lookup(self.transfer, _transfer)
        self.output = f(self.output_in)
