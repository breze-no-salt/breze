# -*- coding: utf-8 -*-


import numpy as np
import theano

from breze.arch.model.linear import Linear
from breze.learn.base import SupervisedBrezeWrapperBase


class GeneralizedLinearModel(Linear, SupervisedBrezeWrapperBase):

    def __init__(self, n_inpt, n_output,
                 out_transfer='identity', loss='squared',
                 optimizer='lbfgs', batch_size=None,
                 max_iter=1000, verbose=False):
        """Create a GeneralizedLinearModel object.

        :param n_inpt: Input dimensionality of a single input.
        :param n_output: Input dimensionality of a single input.
        :param out_transfer: Either a string pointing to a function in
            ``breze.arch.component.transfer`` or a function taking a theano 2D
            tensor and returning a tensor of the same shape.
        :param loss: Either a string pointing to a function in
            ``breze.arch.component.distance`` or a function taking a theano 2D
            tensor and returning a Theano scalar.
        :param optimizer: Can be either a string or a pair. In any case,
            climin.util.optimizer is used to construct an optimizer. In the case
            of a string, the string is used as an identifier for the optimizer
            which is then instantiated with default arguments. If a pair,
            expected to be (`identifier`, `kwargs`) for more fine control of the
            optimizer.
        :param batch_size: Number of examples per batch when calculing the loss
            and its derivatives. None means to use all samples every time.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(GeneralizedLinearModel, self).__init__(
            n_inpt, n_output, out_transfer, loss)

        self.optimizer = optimizer
        self.batch_size = batch_size

        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None

        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)
