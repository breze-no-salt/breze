# -*- coding: utf-8 -*-

"""Module for learning various types of multilayer perceptrons."""


import itertools 

import climin
import numpy as np
import theano.tensor as T

from breze.model.neural import MultiLayerPerceptron

from brummlearn.base import SupervisedBrezeWrapperBase


class Mlp(MultiLayerPerceptron, SupervisedBrezeWrapperBase):

    def __init__(self, n_inpt, n_hiddens, n_output, 
                 hidden_transfers, out_transfer, loss,
                 optimizer='lbfgs',
                 max_iter=1000, verbose=False):
        """Create an Mlp object.

        This implementation uses a stack of affine mappings with a subsequent
        non linearity.

        :param n_inpt: Dimensionality of a single input.
        :param n_hiddens: List of integers, where each integer specifies the
            size of that layer.
        :param n_output: Dimensionality of a single output.
        :param hidden_transfers: List of transfer functions. A transfer function
            is either a string pointing to a function in
            ``breze.component.transfer`` or a function taking a theano 2D tensor
            and returning a tensor of the same shape.
        :param optimizer: String identifying the optimizer to use. Can only be
            ``lbfgs`` for now.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(Mlp, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer,
            loss)

        self.optimizer = optimizer
        if optimizer not in ('lbfgs'):
            raise ValueError('only lbfgs allowed as optimizer for now')

        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape)
