# -*- coding: utf-8 -*-


import numpy as np
import theano

from breze.arch.model.linear import Linear
from breze.learn.base import SupervisedBrezeWrapperBase


class GeneralizedLinearModel(Linear, SupervisedBrezeWrapperBase):
    """Class to represent a linear model."""

    def __init__(self, n_inpt, n_output,
                 out_transfer='identity', loss='squared',
                 optimizer='lbfgs', batch_size=None,
                 max_iter=1000, verbose=False):
        """Create a GeneralizedLinearModel object.

        Parameters
        ---------

        n_inpt : integer
            Input dimensionality of a single input.

        n_output : integer
            Input dimensionality of a single input.

        out_transfer : strong or function
            Either a string pointing to a function in
            ``breze.arch.component.transfer`` or a function taking a theano 2D
            tensor and returning a tensor of the same shape.

        loss : string or function
            Either a string pointing to a function in
            ``breze.arch.component.loss`` or a function taking a theano 2D
            tensor and returning a Theano scalar.

        optimizer : string or pair, optional [default: 'lbfgs']
            Can be either a string or a pair. In any case,
            ``climin.util.optimizer`` is used to construct an optimizer. In the
            case of a string, the string is used as an identifier for the
            optimizer which is then instantiated with default arguments. If a
            pair, expected to be ``(identifier, kwargs)`` for more fine control
            of the optimizer.

        batch_size : integer
            Number of examples per batch when calculing the loss and its
            derivatives. None means to use all samples every time.

        max_iter : integer
            Maximum number of optimization iterations to perform.

        verbose : boolean
            Flag indicating whether to print out information during fitting.
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
