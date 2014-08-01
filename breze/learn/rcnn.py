from breze.arch.component.misc import project_into_l2_ball

__author__ = 'apuigdom'
# -*- coding: utf-8 -*-

"""Module for learning recurrent convolutional neural networks."""

import numpy as np
import theano
import theano.tensor as T

from climin.util import minibatches

from breze.arch.model.neural import rcnn
from breze.arch.util import ParameterSet, Model
from breze.learn.base import SupervisedBrezeWrapperBase

# TODO check docstrings


class Rcnn(Model, SupervisedBrezeWrapperBase):
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
    sample_dim = 1, 1


    def __init__(self, n_inpt, n_hidden_conv, n_hidden_full, n_output,
                 hidden_conv_transfers, hidden_full_transfers, out_transfer,
                 loss, image_height=None, image_width=None, n_image_channel=1,
                 pool_shapes=None, filter_shapes=None,
                 optimizer='lbfgs', batch_size=1, max_iter=1000, recurrent_layers=None,
                 verbose=False, n_time_steps=1, weights=False, p_dropout_inpt=False,
                 p_dropout_conv=False, p_dropout_full=False, clipping=False):

        if filter_shapes is None:
            filter_shapes = [[5, 5] for _ in range(len(n_hidden_conv))]
        if pool_shapes is None:
            pool_shapes = [[2, 2] for _ in range(len(n_hidden_conv))]
        if len(n_hidden_conv) != len(hidden_conv_transfers):
            raise ValueError('n_hidden_conv and hidden_conv_transfers have to '
                             'be of the same length')
        if len(n_hidden_full) != len(hidden_full_transfers):
            raise ValueError('n_hidden_full and hidden_full_transfers have to '
                             'be of the same length')
        if len(filter_shapes) != len(n_hidden_conv):
            raise ValueError('n_hidden_conv and filter_shapes have to '
                             'be of the same length')
        self.batch_size = batch_size
        self.n_time_steps = n_time_steps
        if image_height is None or image_width is None:
            self.n_inpt = (self.n_time_steps, self.batch_size,
                           n_image_channel, 1, n_inpt)
        else:
            self.n_inpt = (self.n_time_steps, self.batch_size,
                           n_image_channel, image_height,
                           image_width)

        if recurrent_layers is None:
            recurrent_layers = [False]*(len(n_hidden_conv)+len(n_hidden_full))

        self.recurrent_layers = recurrent_layers
        self.n_hidden_conv = n_hidden_conv
        self.n_hidden_full = n_hidden_full
        self.n_output = n_output
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_conv = p_dropout_conv
        self.p_dropout_full = p_dropout_full
        self.hidden_conv_transfers = hidden_conv_transfers
        self.hidden_full_transfers = hidden_full_transfers
        self.out_transfer = out_transfer
        self.loss = loss
        self.image_shapes = []
        self.filter_shapes_comp = []
        self.pool_shapes = pool_shapes
        self.filter_shapes = filter_shapes
        self.weights = weights
        self.clipping = clipping
        self._init_image_shapes()
        self._init_filter_shapes()

        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        super(Rcnn, self).__init__()

    def _init_filter_shapes(self):
        self.filter_shapes_comp.append(
            (self.n_hidden_conv[0], self.n_inpt[2], self.filter_shapes[0][0],
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
        image_size = self.n_inpt[-2:]
        self.image_shapes.append(self.n_inpt)
        #input shape = n_time_steps, n_samples, channels, n_frames_to_take, n_output
        zipped = zip(self.n_hidden_conv, self.filter_shapes, self.pool_shapes)
        for n_feature_maps, filter_shape, pool_shape in zipped:
            image_size = [(comp - fs + 1) / ps for comp, fs, ps in
                          zip(image_size, filter_shape, pool_shape)]
            self.image_shapes.append((self.n_time_steps, self.batch_size,
                                      n_feature_maps, image_size[0],
                                      image_size[1]))

        for n_units, rec in zip(self.n_hidden_full, self.recurrent_layers[len(self.n_hidden_conv):]):
            if rec == 'lstm':
                self.image_shapes.append((self.n_time_steps, self.batch_size, 4*n_units))
            else:
                self.image_shapes.append((self.n_time_steps, self.batch_size, n_units))

    def _init_pars(self):
        spec = rcnn.parameters(
            self.n_inpt, self.n_hidden_conv, self.n_hidden_full, self.n_output,
            self.filter_shapes, self.image_shapes, self.recurrent_layers)

        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.tensor3('inpt'),
            'target': T.tensor3('target')
        }
        if self.weights:
            self.exprs['weights'] = T.tensor3('weights')
        else:
            self.exprs['weights'] = None
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
        recurrents = []
        initial_hiddens = []
        states = []
        for i in range(len(self.n_hidden_conv) + len(self.n_hidden_full)):
            rec_attribute = 'recurrent_%i' % i
            ih_attribute = 'initial_hiddens_%i' % i
            s_attribute = 'state_%i' % i
            if hasattr(P, rec_attribute):
                recurrents.append(getattr(P, rec_attribute))
            elif hasattr(P, rec_attribute+'_0'):
                recurrents.append((getattr(P, rec_attribute+'_0'),
                                   getattr(P, rec_attribute+'_1')))
            else:
                recurrents.append(None)
            if hasattr(P, ih_attribute):
                initial_hiddens.append(getattr(P, ih_attribute))
            else:
                initial_hiddens.append(None)
            if hasattr(P, s_attribute):
                states.append(getattr(P, s_attribute))
            else:
                states.append(None)


        ingate_peepholes = []
        outgate_peepholes = []
        forgetgate_peepholes = []

        for i in range(len(self.n_hidden_full)):
            layer = (i+len(self.n_hidden_conv))
            print layer
            attribute = 'ingate_peephole_%i' % layer
            if hasattr(P, attribute):
                ingate_peepholes.append(getattr(P, attribute))
            else:
                ingate_peepholes.append(None)
            attribute = 'outgate_peephole_%i' % layer
            if hasattr(P, attribute):
                outgate_peepholes.append(getattr(P, attribute))
            else:
                outgate_peepholes.append(None)
            attribute = 'forgetgate_peephole_%i' % layer
            if hasattr(P, attribute):
                forgetgate_peepholes.append(getattr(P, attribute))
            else:
                forgetgate_peepholes.append(None)

        self.exprs.update(rcnn.exprs(
            self.exprs['inpt'], self.exprs['target'], P.in_to_hidden, P.hidden_to_out,
            P.out_bias, P.hidden_conv_to_hidden_full, hidden_conv_to_hidden_conv,
            hidden_full_to_hidden_full, hidden_conv_bias,
            hidden_full_bias, self.hidden_conv_transfers,
            self.hidden_full_transfers, self.out_transfer, self.loss,
            self.image_shapes, self.filter_shapes_comp,
            self.pool_shapes, recurrents, initial_hiddens,
            self.exprs['weights'], self.recurrent_layers,
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_conv=self.p_dropout_conv,
            p_dropout_full=self.p_dropout_full,
            ingate_peephole=ingate_peepholes,
            outgate_peephole=outgate_peepholes,
            forgetgate_peephole=forgetgate_peepholes, states=states)
        )


    def _make_loss_functions(self, mode=None, weights=False):
        """Return pair `f_loss, f_d_loss` of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
           matrix of the loss.
        """
        d_loss = self._d_loss()

        if self.clipping:
            d_loss = project_into_l2_ball(d_loss, self.clipping)

        args = list(self.data_arguments)
        if weights:
            args += ['weights']
        f_loss = self.function(args, 'loss', explicit_pars=True, mode=mode)
        f_d_loss = self.function(args, d_loss, explicit_pars=True, mode=mode)
        return f_loss, f_d_loss

    def apply_minibatches_function(self, f, X, Z, weights=None):
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
        if weights is None:
            data = [minibatches(i, self.batch_size, d)
                    for i, d in zip([X, Z], self.sample_dim)]
            total = [f(*element) for element in zip(data[0], data[1])]
        else:
            data = [minibatches(i, self.batch_size, d)
                    for i, d in zip([X, Z, weights],
                                    list(self.sample_dim) + [self.sample_dim[0]])]
            total = [f(*element) for element in zip(data[0], data[1], data[2])]
        return sum(total) / float(len(total))

    def loss_(self, X, Z, weights=None):
        """Override the loss function.

        :param X: numpy array
            Input
        :param Z: numpy array
            Target
        :returns: The loss of the network with respect to the input and the target.
        """
        return self.apply_minibatches_function(super(Rcnn, self).loss, X, Z, weights)

    def score(self, X, Z, weights=None):
        """Override the score function.

        :param X: numpy array
            Input
        :param Z: numpy array
            Target
        :returns: The score of the network with respect to the input and the target.
        """
        return self.apply_minibatches_function(super(Rcnn, self).score, X, Z, weights)

    def predict(self, X):
        """Override the predict function.

        :param X: numpy array
            Input
        :returns: The predictions of the network.
        """
        data = minibatches(X, self.batch_size, 0)
        total = np.concatenate([super(Rcnn, self).predict(element) for element in data], axis=0)
        return total

    def get_deterministic_rcnn(self, loss=None, get_lstm=False):
        if loss is None:
            loss = self.loss

        recurrent_layers = self.recurrent_layers
        if get_lstm:
            recurrent_layers = ['lstm' if i else None for i in self.recurrent_layers]

        return Rcnn(1, self.n_hidden_conv, self.n_hidden_full, self.n_output,
                 self.hidden_conv_transfers, self.hidden_full_transfers, self.out_transfer,
                 loss, image_height=self.n_inpt[-2], image_width=self.n_inpt[-1],
                 n_image_channel=self.n_inpt[-3],
                 pool_shapes=self.pool_shapes, filter_shapes=self.filter_shapes,
                 optimizer=self.optimizer, batch_size=self.batch_size,
                 max_iter=self.max_iter, recurrent_layers=recurrent_layers,
                 verbose=False, n_time_steps=self.n_time_steps, weights=self.weights, p_dropout_inpt=False,
                 p_dropout_conv=False, p_dropout_full=False)