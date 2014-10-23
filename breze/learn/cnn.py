# -*- coding: utf-8 -*-

"""Module for learning convolutional neural networks."""


import theano
import theano.tensor as T

import numpy as np

from breze.arch.model.neural import cnn
from breze.arch.util import ParameterSet, Model
from breze.learn.base import SupervisedBrezeWrapperBase

from climin.util import minibatches
from climin.initialize import randomize_normal


class Cnn(Model, SupervisedBrezeWrapperBase):
    """Cnn class.

    Parameters
    ----------

    n_inpt : Integer
        Dimensionality of a single input.

    n_hidden_conv : List of integers
        List of ``k`` integers, where ``k`` is the number of convolutional
        layers, considering one convolutional layer as the convolution operation
        followed by the pooling operation.
        Each integer gives the number of feature maps of the corresponding
        layer.

    n_hidden_full : List of integers
        List of ``k`` integers, where ``k`` is the number of fully connected
        layers.
        Each gives the size of the corresponding layer.

    n_output : Integer
        Dimensionality of a single output.

    hidden_conv_transfers : List of strings or callables
        Transfer functions for each of the  convolutional layers.
        Can be either a string which is then used to look up a transfer
        function in ``breze.component.transfer`` or a function that given
        a Theano tensor returns a tensor of the same shape.

    hidden_full_transfers : List of strings or callables
        Transfer functions for each of the fully connected layers.
        Can be either a string which is then used to look up a transfer
        function in ``breze.component.transfer`` or a function that given
        a Theano tensor returns a tensor of the same shape.

    out_transfer : String or function
        Either a string to look up a function in ``breze.component.transfer`` or
        a function that given a Theano tensor returns a tensor of the same
        shape.

    loss : String or callable
        Either a string to look up a function in ``breze.component.loss`` or
        a function that, given an input and target Theano tensors, outputs
        a real number.

    image_height : Integer, None
        Indicates the size of the last dimension of the when the input is
        to be considered as 2 or 3-dimensional.

    image_width : Integer, None
        Indicates the size of the second from last dimension when the input is
        to be considered 2 or 3-dimensional.

    n_image_channel : Integer
        Number of channels of the image.

    optimizer : String, pair
        Argument is passed to ``climin.util.optimizer`` to construct an
        optimizer.

    batch_size : Integer, None
        Number of examples per batch when calculating the loss
        and its derivatives. None means to use a single sample every time.

    max_iter : int
        Maximum number of optimization iterations to perform. Only respected
        during``.fit()``, not ``.iter_fit()``.

    verbose : boolean
        Flag indicating whether to print out information during fitting.

    pool_shapes : List of integer tuples
        Size of the pool used for downsampling each convolutional layer.

    pool_strides : List of integer tuples
        Separation between patches for each convolutional layer.
        By default, it is defined to be the same as pool_shapes
        (non-overlapping pooling).

    filter_shapes : List of integer tuples
        Size of the filter (height, width) for the convolutional operation
        for each convolutional layer.

    padding : List of integers
        Pad to add to the input of each convolutional layer.

    lrnorm : List of triples of floats, None
        Parameters (alpha, beta, size of patch) of the local response
        normalization of each convolutional layer. If None, no local response
        normalization is performed. It also can be None for some layers.
        E.g. In a CNN with 2 convolutional layers [(2,2,4), None] would do a
        local response normalization for the first convolutional layer only,
        with alpha = 2, beta = 3, and patch size = 4.

    init_weights_stdev : List of floats
        Standard deviation of the normal distribution used for the
        initialization of the parameters in the model in a per-layer basis.

    init_biases_stdev : List of floats
        Standard deviation of the normal distribution used for the
        initialization of the biases in the model in a per-layer basis.
    """

    def __init__(self, n_inpt, n_hidden_conv, n_hidden_full, n_output,
                 hidden_conv_transfers, hidden_full_transfers, out_transfer,
                 loss, image_height=None, image_width=None, n_image_channel=1,
                 optimizer='lbfgs', batch_size=1, max_iter=1000,
                 verbose=False, pool_shapes=None, pool_strides=None,
                 filter_shapes=None, padding=None, lrnorm=None,
                 init_weights_stdev=None, init_biases_stdev=None):

        n_conv = len(n_hidden_conv)
        n_full = len(n_hidden_full)
        n_all = n_conv + n_full + 1
        self.batch_size = batch_size
        self.padding = [0]*n_conv if padding is None else padding
        self.padding += [0]

        self.lrnorm = lrnorm
        if self.lrnorm is not None:
            if not isinstance(self.lrnorm[0], list):
                self.lrnorm = [self.lrnorm] * n_conv
        else:
            self.lrnorm = [None] * n_conv

        self.filter_shapes = filter_shapes
        if self.filter_shapes is None:
            self.filter_shapes = [[5, 5]] * n_conv

        self.pool_shapes = pool_shapes
        if self.pool_shapes is None:
            self.pool_shapes = [[2, 2]] * n_conv

        self.pool_strides = pool_strides
        if self.pool_strides is None:
            self.pool_strides = self.pool_shapes

        self.init_weigths_stdev = init_weights_stdev
        if init_weights_stdev is None:
            self.init_weigths_stdev = [0.01] * n_all

        self.init_biases_stdev = init_biases_stdev
        if init_biases_stdev is None:
            self.init_biases_stdev = [0] * n_all

        if image_height is None or image_width is None:
            self.n_inpt = (self.batch_size, n_image_channel, n_inpt, 1)
        else:
            self.n_inpt = (self.batch_size, n_image_channel,
                           image_height, image_width)

        if len(hidden_conv_transfers) != n_conv:
            raise ValueError('n_hidden_conv and hidden_conv_transfers have to '
                             'be of the same length')
        if len(hidden_full_transfers) != n_full:
            raise ValueError('n_hidden_full and hidden_full_transfers have to '
                             'be of the same length')
        if len(self.filter_shapes) != n_conv:
            raise ValueError('n_hidden_conv and filter_shapes have to '
                             'be of the same length')
        if len(self.init_weigths_stdev) != n_all:
            raise ValueError('n_hidden_conv and init_weigths_stdev have to '
                             'be of the same length')
        if len(self.init_biases_stdev) != n_all:
            raise ValueError('n_hidden_conv and init_biases_stdev have to '
                             'be of the same length')
        if len(self.pool_shapes) != n_conv:
            raise ValueError('n_hidden_conv and pool_shapes have to '
                             'be of the same length')
        if len(self.pool_strides) != n_conv:
            raise ValueError('n_hidden_conv and pool_strides have to '
                             'be of the same length')

        self.n_hidden_conv = n_hidden_conv
        self.n_hidden_full = n_hidden_full
        self.n_output = n_output
        self.hidden_conv_transfers = hidden_conv_transfers
        self.hidden_full_transfers = hidden_full_transfers
        self.out_transfer = out_transfer
        self.loss = loss
        self.image_shapes = []
        self.filter_shapes_comp = []
        self.pool_shifts = []
        self._init_image_shapes()
        self._init_filter_shapes()
        self._init_pool_shifts()
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        super(Cnn, self).__init__()

    def _init_filter_shapes(self):
        """Calculation of the size of the filters used in the convolutional
        layers in advance. The convolution operation of Theano needs to know
        these shapes at compilation time.
        """
        self.filter_shapes_comp.append(
            (self.n_hidden_conv[0], self.n_inpt[1], self.filter_shapes[0][0],
             self.filter_shapes[0][1]))
        zipped = zip(self.n_hidden_conv[:-1], self.n_hidden_conv[1:],
                     self.filter_shapes[1:])
        for inlayer, outlayer, filter_shape in zipped:
            self.filter_shapes_comp.append(
                (outlayer, inlayer, filter_shape[0], filter_shape[1]))

    def _init_image_shapes(self):
        """Calculation of the size of the inputs used in the convolutional
        layers in advance. The convolution operation of Theano needs to know
        these shapes at compilation time.
        """
        image_size = (self.n_inpt[2]+2*self.padding[0],
                      self.n_inpt[3]+2*self.padding[0])
        self.image_shapes.append(self.n_inpt[:2]+image_size)
        zipped = zip(self.n_hidden_conv, self.filter_shapes,
                     self.pool_shapes, self.pool_strides, self.padding[1:])
        for n_feat_maps, filter_shape, pool_shape, pool_stride, pad in zipped:
            sub_zipped = zip(image_size, filter_shape, pool_shape, pool_stride)
            image_size = []
            for comp, fs, psh, pst in sub_zipped:
                image_size.append(2*pad + 1 + (comp - fs + 1 - psh) / pst)
            self.image_shapes.append((self.batch_size, n_feat_maps,
                                      image_size[0], image_size[1]))

    def _init_pars(self):
        last_image_shape = self.image_shapes[-1]
        resulting_image_size = last_image_shape[-1] * last_image_shape[-2]

        spec = cnn.parameters(
            self.n_inpt, self.n_hidden_conv, self.n_hidden_full, self.n_output,
            resulting_image_size, self.filter_shapes)

        self.parameters = ParameterSet(**spec)

    def _init_pool_shifts(self):
        """Initialization of the shifted images we need to obtain for every
        pooling layer in order to apply the pooling with the specified stride
        in the constructor.
        """
        for sh, st in zip(self.pool_shapes, self.pool_strides):
            pool_shift = [[0], [0]]
            i, j = st
            while (i % sh[0]) != 0:
                pool_shift[0].append(i)
                i += st[0]
            while (j % sh[1]) != 0:
                pool_shift[1].append(j)
                j += st[1]
            self.pool_shifts.append(pool_shift)

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
            'target': T.matrix('target')
        }
        P = self.parameters
        hidden_conv_to_hidden_conv = [P['hidden_conv_to_hidden_conv_%i' % i]
                                      for i in range(len(self.n_hidden_conv)
                                                     - 1)]
        hidden_full_to_hidden_full = [P['hidden_full_to_hidden_full_%i' % i]
                                      for i in range(len(self.n_hidden_full)
                                                     - 1)]
        hidden_conv_bias = [getattr(P, 'hidden_conv_bias_%i' % i)
                            for i in range(len(self.n_hidden_conv))]
        hidden_full_bias = [getattr(P, 'hidden_full_bias_%i' % i)
                            for i in range(len(self.n_hidden_full))]
        self.exprs.update(cnn.exprs(
            self.exprs['inpt'], self.exprs['target'],
            P.in_to_hidden, P.hidden_to_out,
            P.out_bias, P.hidden_conv_to_hidden_full,
            hidden_conv_to_hidden_conv, hidden_full_to_hidden_full,
            hidden_conv_bias, hidden_full_bias, self.hidden_conv_transfers,
            self.hidden_full_transfers, self.out_transfer, self.loss,
            self.image_shapes, self.filter_shapes_comp, self.n_inpt,
            self.pool_shapes, self.pool_shifts, self.padding, self.lrnorm))

    def draw_params(self, seed=314):
        """Initialization of the parameters of the model. These are drawn
        from a normal distribution on a per-layer basis.

        Parameters
        ----------

        seed : Integer
            Random seed to use for the random number generator.
        """
        rng = np.random.RandomState(seed)
        self.parameters.data[:] = rng.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)
        weight_order = ['in_to_hidden']
        bias_order = ['hidden_conv_bias_0']
        for i in np.arange(1, len(self.n_hidden_conv)):
            weight_order.append('hidden_conv_to_hidden_conv_%i' % (i-1))
            bias_order.append('hidden_conv_bias_%i' % i)
        weight_order.append('hidden_conv_to_hidden_full')
        bias_order.append('hidden_full_bias_0')
        for i in range(1, len(self.n_hidden_full)):
            weight_order.append('hidden_full_to_full_conv_%i' % (i-1))
            bias_order.append('hidden_conv_bias_%i' % i)
        weight_order.append('hidden_to_out')
        bias_order.append('out_bias')
        for i, (weight, bias) in enumerate(zip(weight_order, bias_order)):
            if self.init_weigths_stdev[i] == 0:
                weight_data = np.zeros(self.parameters[weight].shape)
            else:
                weight_data = rng.normal(0, self.init_weigths_stdev[i],
                                         self.parameters[weight].shape)
            self.parameters[weight] = weight_data
            if self.init_biases_stdev[i] == 0:
                bias_data = np.zeros(self.parameters[bias].shape)
            else:
                bias_data = rng.normal(0, self.init_biases_stdev[i],
                                       self.parameters[bias].shape)
            self.parameters[bias] = bias_data

    def apply_minibatches_function(self, f, X, Z):
        """Apply a function to batches of the input.

        The convolutional neural networks class needs the input to be the same
        size as the batch size. This function slices the input so that it
        can be processed correctly.
        If the batch size is not a divisor of the input size, an exception is
        raised.

        Parameters
        ----------

        f : Callable
            Function to use for all the batches.

        X : Numpy array
            Input of the function

        Z : numpy array
            Target of the function

        Returns
        -------

        exprs : The average of the results of the function over all the batches.
        """
        data = [minibatches(i, self.batch_size, d)
                for i, d in zip([X, Z], self.sample_dim)]
        if theano.config.device == 'gpu':
            total = [f(*element).asndarray()
                     for element in zip(data[0], data[1])]
        else:
            total = [f(*element) for element in zip(data[0], data[1])]
        return sum(total)/float(len(total))

    def loss_(self, X, Z):
        """Override the loss function.

        Parameters
        ----------

        X : Numpy array
            Input of the function

        Z : numpy array
            Target of the function

        Returns
        -------

        exprs : The loss of the network with respect to the input and
        the target.
        """
        return self.apply_minibatches_function(super(Cnn, self).loss, X, Z)

    def score(self, X, Z):
        """Override the score function.

        Parameters
        ----------

        X : Numpy array
            Input of the function

        Z : numpy array
            Target of the function

        Returns
        -------

        exprs : The score of the network with respect to the input and
        the target.
        """
        return self.apply_minibatches_function(super(Cnn, self).score, X, Z)

    def predict(self, X):
        """Override the predict function.

        Parameters
        ----------

        X : Numpy array
            Input of the function

        Returns
        -------

        exprs : The predictions of the network
        """
        data = minibatches(X, self.batch_size, 0)
        total = np.concatenate([super(Cnn, self).predict(element)
                                for element in data], axis=0)
        return total
