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
        super(Mlp, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer,
            loss)
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape)
