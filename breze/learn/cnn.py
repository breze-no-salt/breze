# -*- coding: utf-8 -*-

"""Module for learning convolutional neural networks."""

import numpy as np
import theano
import theano.tensor as T

from breze.arch.model.neural import cnn
from breze.arch.util import ParameterSet, Model
from breze.learn.base import SupervisedBrezeWrapperBase
from breze.learn.data import minibatches

# TODO check docstrings


class Cnn(Model, SupervisedBrezeWrapperBase):
    """Cnn class.

    Parameters
    ----------

    n_inpt : integer
        Dimensionality of a single input.

    n_hidden_conv : list of integers
        List of ``k`` integers, where ``k`` is the number of convolutional
        layers, considering one convolutional layer as the convolution operation
        followed by the pooling operation.
        Each integer gives the number of feature maps of the corresponding layer.

    n_hidden_full : list of integers
        List of ``k`` integers, where ``k`` is the number of fully connected
        layers.
        Each gives the size of the corresponding layer.

    n_output : integer
        Dimensionality of a single output.

    hidden_conv_transfers : list, each item either string or function
        Transfer functions for each of the  convolutional layers.
        Can be either a string which is then used to look up a transfer
        function in ``breze.component.transfer`` or a function that given
        a Theano tensor returns a tensor of the same shape.

    hidden_full_transfers : list, each item either string or function
        Transfer functions for each of the fully connected layers.
        Can be either a string which is then used to look up a transfer
        function in ``breze.component.transfer`` or a function that given
        a Theano tensor returns a tensor of the same shape.

    out_transfer : string or function
        Either a string to look up a function in ``breze.component.transfer`` or
        a function that given a Theano tensor returns a tensor of the same
        shape.

    optimizer : string, pair
        Argument is passed to ``climin.util.optimizer`` to construct an
        optimizer.

    batch_size : integer, None
        Number of examples per batch when calculating the loss
        and its derivatives. None means to use a single sample every time.

    max_iter : int
        Maximum number of optimization iterations to perform. Only respected
        during``.fit()``, not ``.iter_fit()``.

    verbose : boolean
        Flag indicating whether to print out information during fitting.

    pool_size : integer tuple
        Size of the pool used for downsampling in the convolutional layers.

    filter_shape : integer tuple
        Size of the filter (height, width) for the convolutional operation.
    """

    def __init__(self, n_inpt, n_hidden_conv, n_hidden_full, n_output,
                 hidden_conv_transfers, hidden_full_transfers, out_transfer,
                 loss, image_height=None, image_width=None, n_image_channel=1,
                 pool_size=(2, 2), filter_shapes=None,
                 optimizer='lbfgs', batch_size=None, max_iter=1000,
                 verbose=False):

        if filter_shapes is None:
            filter_shapes = [[5, 5] for _ in range(len(n_hidden_conv))]
        if len(n_hidden_conv) != len(hidden_conv_transfers):
            raise ValueError('n_hidden_conv and hidden_conv_transfers have to '
                             'be of the same length')
        if len(n_hidden_full) != len(hidden_full_transfers):
            raise ValueError('n_hidden_full and hidden_full_transfers have to '
                             'be of the same length')
        if len(filter_shapes) != len(n_hidden_conv):
            raise ValueError('n_hidden_conv and filter_shapes have to '
                             'be of the same length')
        self.batch_size = 1 if batch_size is None else batch_size
        if image_height is None or image_width is None:
            self.n_inpt = (self.batch_size, n_image_channel, n_inpt, 1)
        else:
            self.n_inpt = (self.batch_size, n_image_channel,
                           image_height, image_width)
        self.n_hidden_conv = n_hidden_conv
        self.n_hidden_full = n_hidden_full
        self.n_output = n_output
        self.hidden_conv_transfers = hidden_conv_transfers
        self.hidden_full_transfers = hidden_full_transfers
        self.out_transfer = out_transfer
        self.loss = loss
        self.image_shapes = []
        self.filter_shapes_comp = []
        self.pool_size = pool_size
        self.filter_shapes = filter_shapes
        self._init_image_shapes()
        self._init_filter_shapes()

        self.optimizer = optimizer
        self.batch_size = batch_size

        self.max_iter = max_iter
        self.verbose = verbose

        super(Cnn, self).__init__()

    def _init_filter_shapes(self):
        self.filter_shapes_comp.append(
            (self.n_hidden_conv[0], self.n_inpt[1], self.filter_shapes[0][0],
             self.filter_shapes[0][1]))
        zipped = zip(self.n_hidden_conv[:-1], self.n_hidden_conv[1:],
                     self.filter_shapes[1:])
        for inlayer, outlayer, filter_shape in zipped:
            self.filter_shapes_comp.append(
                (outlayer, inlayer, filter_shape[0], filter_shape[1]))

    def _init_image_shapes(self):
        if len(self.n_hidden_conv) == 0:
            raise ValueError('If you are not going to use convolutional layers,'
                             ' please use MultilayerPerceptron.')
        image_size = [self.n_inpt[2], self.n_inpt[3]]
        self.image_shapes.append(self.n_inpt)
        zipped = zip(self.n_hidden_conv, self.filter_shapes)
        for n_feature_maps, filter_shape in zipped:
            image_size = [(comp - fs + 1) / ps for comp, fs, ps in
                          zip(image_size, filter_shape, self.pool_size)]
            self.image_shapes.append((self.batch_size, n_feature_maps,
                                      image_size[0], image_size[1]))

    def _init_pars(self):
        last_image_shape = self.image_shapes[-1]
        resulting_image_size = last_image_shape[-1] * last_image_shape[-2]

        spec = cnn.parameters(
            self.n_inpt, self.n_hidden_conv, self.n_hidden_full, self.n_output,
            resulting_image_size, self.filter_shapes)

        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
            'target': T.matrix('target')
        }
        P = self.parameters

        hidden_conv_to_hidden_conv = [getattr(P,
                                              'hidden_conv_to_hidden_conv_%i' % i)
                                      for i in range(len(self.n_hidden_conv) - 1)]
        hidden_full_to_hidden_full = [getattr(P,
                                              'hidden_full_to_hidden_full_%i' % i)
                                      for i in range(len(self.n_hidden_full) - 1)]
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
            self.image_shapes[:-1], self.filter_shapes_comp, self.n_inpt,
            self.pool_size))

    # TODO move this somewhere else
    def sample_conv_weights(self, seed=23455):
        init_vals = []
        rng = np.random.RandomState(seed)
        # init in_to_hidden
        fan_in = np.prod(self.filter_shapes[0][1:])
        fan_out = (self.filter_shapes[0][0] * np.prod(self.filter_shapes[0][2:]) /
                   np.prod(self.pool_size))
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        init_vals.append(('in_to_hidden', rng.uniform(low=-w_bound, high=w_bound, size=self.filter_shapes[0])))
        # init hidden_conv_to_hidden_conv_%i
        for i in np.arange(len(self.n_hidden_conv) - 1):
            fan_in = np.prod(self.filter_shapes[i + 1][1:])
            fan_out = (self.filter_shapes[i + 1][0] * np.prod(self.filter_shapes[i + 1][2:]) /
                       np.prod(self.pool_size))
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            init_vals.append(('hidden_conv_to_hidden_conv_%i' % i,
                              rng.uniform(low=-w_bound, high=w_bound,
                                          size=self.filter_shapes[i + 1])))

        return init_vals

    def apply_minibatches_function(self, f, X, Z):
        """Apply a function to batches of the input.

        The convolutional neural networks class needs the input to be the same
        size as the batch size. This function slices the input so that it can be
        processed correctly.
        If the batch size is not a divisor of the input size, an exception is
        raised.

        :param f: theano function
            Function to use for all the batches.

        :param X: numpy array
            Input of the function

        :param Z: numpy array
            Target of the function

        :returns: The average of the results of the function over all the batches.
        """
        data = [minibatches(i, self.batch_size, d) for i, d in zip([X, Z], self.sample_dim)]
        total = [f(*element) for element in zip(data[0], data[1])]
        return sum(total) / float(len(total))

    def loss_(self, X, Z):
        """Override the loss function.

        :param X: numpy array
            Input
        :param Z: numpy array
            Target
        :returns: The loss of the network with respect to the input and the target.
        """
        return self.apply_minibatches_function(super(Cnn, self).loss, X, Z)

    def score(self, X, Z):
        """Override the score function.

        :param X: numpy array
            Input
        :param Z: numpy array
            Target
        :returns: The score of the network with respect to the input and the target.
        """
        return self.apply_minibatches_function(super(Cnn, self).score, X, Z)

    def predict(self, X):
        """Override the predict function.

        :param X: numpy array
            Input
        :returns: The predictions of the network.
        """
        data = minibatches(X, self.batch_size, 0)
        if theano.config.device == 'gpu':
            raise NotImplementedError(
                'prediction not possible on gpu with conv net yet, please implement :)')
        total = np.concatenate([super(Cnn, self).predict(element) for element in data], axis=0)
        return total
