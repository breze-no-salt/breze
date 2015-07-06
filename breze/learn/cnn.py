# -*- coding: utf-8 -*-

"""Module for learning convolutional neural networks."""


import numpy as np

from theano import tensor as T

from breze.arch.construct.simple import SupervisedLoss
from breze.arch.construct import neural
from breze.arch.util import ParameterSet
from breze.learn.base import SupervisedModel


class SimpleCnn2d(SupervisedModel):

    def __init__(self, image_height, image_width, n_channel,
                 n_hiddens, filter_shapes, n_output,
                 hidden_transfers, out_transfer,
                 loss,
                 optimizer='lbfgs', batch_size=1, max_iter=1000,
                 verbose=False):
        self.image_height = image_height
        self.image_width = image_width
        self.n_channel = n_channel
        self.n_hiddens = n_hiddens
        self.filter_shapes = filter_shapes
        self.hidden_transfers = hidden_transfers
        self.n_output = n_output
        self.out_transfer = out_transfer
        self.loss_ident = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose

        self._init_exprs()

    def _init_exprs(self):
        inpt = T.tensor4('inpt')
        inpt.tag.test_value = np.zeros((
            2, self.n_channel, self.image_height, self.image_width))
        target = T.matrix('target')
        target.tag.test_value = np.zeros((
            2, self.n_output))
        parameters = ParameterSet()

        self.cnn = neural.SimpleCnn2d(
            inpt,
            self.image_height, self.image_width,
            self.n_channel, self.n_hiddens, self.filter_shapes, self.n_output,
            self.hidden_transfers, self.out_transfer,
            declare=parameters.declare)

        if self.imp_weight:
            imp_weight = T.matrix('imp_weight')
        else:
            imp_weight = None

        self.loss_layer = SupervisedLoss(
            target, self.cnn.output, loss=self.loss_ident,
            imp_weight=imp_weight,
            declare=parameters.declare,
        )

        SupervisedModel.__init__(self, inpt=inpt, target=target,
                                 output=self.cnn.output,
                                 loss=self.loss_layer.total,
                                 parameters=parameters)
        self.exprs['imp_weight'] = imp_weight


class Lenet(SupervisedModel):

    def __init__(self, image_height, image_width, n_channel,
                 n_hiddens_conv, filter_shapes, pool_shapes,
                 n_hiddens_full,
                 n_output,
                 hidden_transfers_conv, hidden_transfers_full,
                 out_transfer,
                 loss,
                 optimizer='lbfgs', batch_size=1, max_iter=1000,
                 verbose=False):
        self.image_height = image_height
        self.image_width = image_width
        self.n_channel = n_channel
        self.n_hiddens_conv = n_hiddens_conv
        self.n_hiddens_full = n_hiddens_full
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.hidden_transfers_conv = hidden_transfers_conv
        self.hidden_transfers_full = hidden_transfers_full
        self.n_output = n_output
        self.out_transfer = out_transfer
        self.loss_ident = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose

        self._init_exprs()

    def _init_exprs(self):
        inpt = T.tensor4('inpt')
        inpt.tag.test_value = np.zeros((
            2, self.n_channel, self.image_height, self.image_width))
        target = T.matrix('target')
        target.tag.test_value = np.zeros((
            2, self.n_output))
        parameters = ParameterSet()

        self.lenet = neural.Lenet(
            inpt,
            self.image_height,
            self.image_width,
            self.n_channel,
            self.n_hiddens_conv,
            self.filter_shapes,
            self.pool_shapes,
            self.n_hiddens_full,
            self.hidden_transfers_conv,
            self.hidden_transfers_full,
            self.n_output,
            self.out_transfer,
            declare=parameters.declare,
        )

        if self.imp_weight:
            imp_weight = T.matrix('imp_weight')
        else:
            imp_weight = None

        self.loss_layer = SupervisedLoss(
            target, self.lenet.output, loss=self.loss_ident,
            imp_weight=imp_weight,
            declare=parameters.declare,
        )

        SupervisedModel.__init__(self, inpt=inpt, target=target,
                                 output=self.lenet.output,
                                 loss=self.loss_layer.total,
                                 parameters=parameters)
        self.exprs['imp_weight'] = imp_weight
