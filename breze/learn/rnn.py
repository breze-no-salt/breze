# -*- coding: utf-8 -*-

"""Module for learning various types of recurrent networks."""


import numpy as np
import theano
import theano.tensor as T

from breze.arch.component.misc import project_into_l2_ball
from breze.arch.component.common import supervised_loss, unsupervised_loss
from breze.arch.component.varprop.common import supervised_loss as varprop_supervised_loss
from breze.arch.component.varprop import loss as vp_loss
from breze.arch.model.varprop import rnn as varprop_rnn
from breze.arch.model.rnn import rnn, lstm
from breze.arch.util import ParameterSet, Model, lookup
from breze.learn.base import (
    SupervisedBrezeWrapperBase, UnsupervisedBrezeWrapperBase)
from breze.arch.construct.base import SupervisedStack
from breze.arch.construct.layer import simple, sequential
from breze.arch.construct.layer.varprop import (
        simple as vp_simple, sequential as vp_sequential)
#varprop import rnn as varprop_rnn


# TODO check docstrings (e.g. loss is wrong)


class BaseRnn(object):
    """Base class for RNNs.

    Parameters
    ----------

    n_inpt : integer
        Number of inputs per time step to the network.

    n_hidden : integer
        Size of the hidden state.

    n_output : integer
        Size of the output of the network.

    hidden_transfers : list of string or functions
        Transfer functions to use for the network. Each item can either be (a) a
        string and reference a transfer function from
        ``breze.arch.component.transfer`` or (b) a function which takes a theano
        tensor3 and returns a tensor of equal size.

    out_transfer : string or functions
        Output function to use for the network. This can either (a) be a string
        and reference a transfer function from ``breze.arch.component.transfer``
        or (b) a function which takes a theano tensor3 and returns a tensor of equal
        size.

    loss : string or function
        Loss which is going to be optimized. This can either be a string and
        reference a loss function found in ``breze.arch.component.distance`` or
        a function which takes two theano tensors (one being the output of the
        network, the other some target) and returns a theano scalar.

    pooling: string
        One of ``sum``, ``mean``, ``prod``, ``min``, ``max`` or ``None``. If
        not None, the output is pooled over the time dimension, essentially
        making the network return a tensor2 instead of a tensor3.

    gradient_clip : float, optional [default: False]
        If the length of a gradient ever exceeds this value during training,
        the gradient is renormalized to this value.

    imp_weight : boolean
        Flag indicating whether importance weights are used.

    optimizer : string, pair
        Argument is passed to ``climin.util.optimizer`` to construct an
        optimizer.

    batch_size : integer, None
        Number of examples per batch when calculting the loss
        and its derivatives. None means to use all samples every time.

    max_iter : int
        Maximum number of optimization iterations to perform. Only respected
        during``.fit()``, not ``.iter_fit()``.

    verbose : boolean
        Flag indicating whether to print out information during fitting.
    """
    # TODO: document imp_weight, ideally with an example.

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer='identity',
                 loss='squared', pooling=None,
                 gradient_clip=False,
                 optimizer='rprop',
                 batch_size=None,
                 imp_weight=False,
                 max_iter=1000,
                 verbose=False):
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self._loss = loss
        self.pooling = pooling
        self.gradient_clip = gradient_clip
        self.imp_weight = imp_weight
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose

        self._init_layers()

    def _gauss_newton_product(self):
        """Return a theano expression for the product of the networks
        Gauss-Newton matrix with an arbitrary vector."""

        # Shortcuts.
        output_in = self.exprs['output_in']
        loss = self.exprs['loss']
        flat_pars = self.parameters.flat
        p = self.exprs['some-vector'] = T.vector('some-vector')

        Jp = T.Rop(output_in, flat_pars, p)
        HJp = T.grad(T.sum(T.grad(loss, output_in) * Jp),
                     output_in, consider_constant=[Jp])
        Hp = T.grad(T.sum(HJp * output_in),
                    flat_pars, consider_constant=[HJp, Jp])

        return Hp

    def _make_loss_functions(self, mode=None, imp_weight=False):
        """Return pair `f_loss, f_d_loss` of functions.

         - f_loss returns the current loss,
         - f_d_loss returns the gradient of that loss wrt parameters,
           matrix of the loss.
        """
        d_loss = self._d_loss()
        if self.gradient_clip:
            d_loss = project_into_l2_ball(d_loss, self.gradient_clip)

        args = list(self.data_arguments)
        if imp_weight:
            args += ['imp_weight']
        f_loss = self.function(args, 'loss', explicit_pars=True, mode=mode)
        f_d_loss = self.function(args, d_loss, explicit_pars=True, mode=mode)
        return f_loss, f_d_loss


class SupervisedRnn(BaseRnn, SupervisedStack):
    # TODO document

    @property
    def sample_dim(self):
        if self.pooling is None:
            return 1, 1
        else:
            return 1, 0

    def _init_layers(self):
        inpt = T.tensor3('inpt')
        if self.pooling:
            target = T.matrix('target')
            imp_weight = T.matrix('imp_weight') if self.imp_weight else None
        else:
            target = T.tensor3('target')
            imp_weight = T.tensor3('imp_weight') if self.imp_weight else None

        n_incoming = [self.n_inpt] + self.n_hiddens[:-1]
        n_outgoing = self.n_hiddens
        transfers = self.hidden_transfers

        layers = []
        s2s = sequential.SequentialToStatic()
        for n, m, t in zip(n_incoming, n_outgoing, transfers):
            layers += [s2s,
                       simple.AffineNonlinear(n, m),
                       s2s.inverse,
                       sequential.Recurrent(m, t)]
        layers += [
            s2s,
            simple.AffineNonlinear(
                n_outgoing[-1], self.n_output, self.out_transfer),
            s2s.inverse]

        if self.pooling:
            layers.append(sequential.Pooling(self.pooling))

        comp_dim = 1 if self.pooling else 2

        loss = simple.SupervisedLoss(
            self._loss, target, imp_weight=imp_weight, comp_dim=comp_dim)

        SupervisedStack.__init__(self, layers, loss)
        self.forward(inpt)



class UnsupervisedRnn(BaseRnn, UnsupervisedBrezeWrapperBase):
    """Class implementing recurrent neural networks for unsupervised learning.

    The class inherits from breze's RecurrentNetwork class and adds several
    sklearn like methods.
    """

    sample_dim = 1,

    def _init_exprs(self):
        super(UnsupervisedRnn, self)._init_exprs()
        self.exprs.update(unsupervised_loss(self.exprs['output'], self.loss, 2))


class BaseLstmRnn(Model):
    """Base class for LSTM-RNNs.

    Parameters
    ----------

    n_inpt : integer
        Number of inputs per time step to the network.

    n_hidden : integer
        Size of the hidden state.

    n_output : integer
        Size of the output of the network.

    out_transfer : string or functions
        Output function to use for the network. This can either (a) be a string
        and reference a transfer function from ``breze.arch.component.transfer``
        or (b) a function which takes a theano tensor3 and returns a tensor of equal
        size.

    loss : string or function
        Loss which is going to be optimized. This can either be a string and
        reference a loss function found in ``breze.arch.component.distance`` or
        a function which takes two theano tensors (one being the output of the
        network, the other some target) and returns a theano scalar.

    pooling: string
        One of ``sum``, ``mean``, ``prod``, ``min``, ``max`` or ``None``. If
        not None, the output is pooled over the time dimension, essentially
        making the network return a tensor2 instead of a tensor3.

    optimizer : string, pair
        Argument is passed to ``climin.util.optimizer`` to construct an
        optimizer.

    batch_size : integer, None
        Number of examples per batch when calculting the loss
        and its derivatives. None means to use all samples every time.

    gradient_clip : float, optional [default: False]
        If the length of a gradient ever exceeds this value during training,
        the gradient is renormalized to this value.

    max_iter : int
        Maximum number of optimization iterations to perform. Only respected
        during``.fit()``, not ``.iter_fit()``.

    verbose : boolean
        Flag indicating whether to print out information during fitting.
    """

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer='identity',
                 loss='squared', pooling=None,
                 leaky_coeffs=None,
                 gradient_clip=False,
                 optimizer='rprop',
                 batch_size=None,
                 max_iter=1000,
                 verbose=False):
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.loss = loss
        self.pooling = pooling
        self.gradient_clip = gradient_clip
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose

        super(BaseLstmRnn, self).__init__()

    def _init_pars(self):
        spec = lstm.parameters(self.n_inpt, self.n_hiddens, self.n_output)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {'inpt': T.tensor3('inpt')}
        P = self.parameters

        n_layers = len(self.n_hiddens)
        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        recurrents = [getattr(P, 'recurrent_%i' % i)
                      for i in range(n_layers)]
        #initial_hiddens = [getattr(P, 'initial_hiddens_%i' % i)
        #                   for i in range(n_layers)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(n_layers)]

        ingate_peepholes = [getattr(P, 'ingate_peephole_%i' % i)
                            for i in range(n_layers)]
        outgate_peepholes = [getattr(P, 'outgate_peephole_%i' % i)
                             for i in range(n_layers)]
        forgetgate_peepholes = [getattr(P, 'forgetgate_peephole_%i' % i)
                                for i in range(n_layers)]

        self.exprs.update(lstm.exprs(
            self.exprs['inpt'], P.in_to_hidden, hidden_to_hiddens,
            P.hidden_to_out, hidden_biases,
            recurrents, P.out_bias, ingate_peepholes, outgate_peepholes,
            forgetgate_peepholes, self.hidden_transfers, self.out_transfer,
            self.pooling))


class SupervisedLstmRnn(BaseLstmRnn, SupervisedBrezeWrapperBase):
    """Class implementing recurrent neural networks with LSTM cells for
    supervised learning.

    The class inherits from breze's RecurrentNetwork class and adds several
    sklearn like methods.
    """

    sample_dim = 1, 1

    def _init_exprs(self):
        super(SupervisedLstmRnn, self)._init_exprs()
        if self.pooling:
            self.exprs['target'] = T.matrix('target')
        else:
            self.exprs['target'] = T.tensor3('target')
        self.exprs.update(supervised_loss(
            self.exprs['target'], self.exprs['output'], self.loss, 2))


class UnsupervisedLstmRnn(BaseLstmRnn, UnsupervisedBrezeWrapperBase):
    """Class implementing recurrent neural networks with LSTM cells for
    unsupervised learning.

    The class inherits from breze's RecurrentNetwork class and adds several
    sklearn like methods.
    """
    sample_dim = 1,

    def _init_exprs(self):
        super(UnsupervisedLstmRnn, self)._init_exprs()
        self.exprs.update(unsupervised_loss(self.exprs['output'], self.loss, 2))


class SupervisedFastDropoutRnn(BaseRnn, SupervisedStack):

    sample_dim = 1, 1

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer='identity',
                 loss='squared', pooling=None,
                 gradient_clip=False,
                 p_dropout_inpt=.2, p_dropout_hiddens=.5,
                 p_dropout_hidden_to_out=None,
                 imp_weight=False,
                 optimizer='rprop',
                 batch_size=None,
                 max_iter=1000,
                 verbose=False):
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens

        if isinstance(self.p_dropout_hiddens, float):
            self.p_dropout_hiddens = [self.p_dropout_hiddens]
        else:
            self.p_dropout_hiddens = self.p_dropout_hiddens
        if p_dropout_hidden_to_out is None:
            self.p_dropout_hidden_to_out = self.p_dropout_hiddens[-1]
        else:
            self.p_dropout_hidden_to_out = p_dropout_hidden_to_out

        super(SupervisedFastDropoutRnn, self).__init__(
            n_inpt, n_hiddens, n_output,
            hidden_transfers, out_transfer, loss, pooling=pooling,
            gradient_clip=gradient_clip,
            optimizer=optimizer, batch_size=batch_size, max_iter=max_iter,
            verbose=verbose, imp_weight=imp_weight)

    def _init_layers(self):
        inpt = T.tensor3('inpt')
        target = T.tensor3('target')
        imp_weight = None if not self.imp_weight else T.tensor3('imp_weight')

        n_incoming = [self.n_inpt] + self.n_hiddens[:-1]
        n_outgoing = self.n_hiddens
        transfers = self.hidden_transfers
        p_dropouts = self.p_dropout_hiddens

        s2s = sequential.SequentialToStatic()
        layers = [s2s,
                  vp_simple.AugmentVariance(),
                  vp_simple.FastDropout(self.p_dropout_inpt),
                  s2s.inverse]

        for n, m, t, d in zip(n_incoming, n_outgoing, transfers, p_dropouts):
            layers += [s2s,
                       vp_simple.AffineNonlinear(n, m),
                       vp_simple.FastDropout(d),
                       s2s.inverse,
                       vp_sequential.FDRecurrent(m, t, d)]
        layers += [
            s2s,
            vp_simple.FastDropout(self.p_dropout_hidden_to_out),
            vp_simple.AffineNonlinear(
                n_outgoing[-1], self.n_output, self.out_transfer),
            s2s.inverse,
            simple.Concatenate(2)]

        loss_ = lookup(self._loss, vp_loss)
        loss = simple.SupervisedLoss(
            loss_, target, imp_weight=imp_weight, comp_dim=2)

        SupervisedStack.__init__(self, layers, loss)
        self.forward(inpt)
