# -*- coding: utf-8 -*-

"""Module for learning various types of recurrent networks."""


import climin.initialize
import numpy as np
import theano.tensor as T

from breze.arch.component.misc import project_into_l2_ball
from breze.arch.component.varprop import loss as vp_loss
from breze.arch.util import ParameterSet, lookup
from breze.learn.base import SupervisedModel
from breze.arch.construct import simple, neural


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
        or (b) a function which takes a theano tensor3 and returns a tensor of
        equal size.

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
        self.loss_ident = loss
        self.pooling = pooling
        self.gradient_clip = gradient_clip
        self.imp_weight = imp_weight
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose

        self._init_exprs()

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


class SupervisedRnn(BaseRnn, SupervisedModel):
    # TODO document

    @property
    def sample_dim(self):
        if self.pooling is None:
            return 1, 1
        else:
            return 1, 0

    def _init_exprs(self):
        inpt = T.tensor3('inpt')
        if self.pooling:
            target = T.matrix('target')
            imp_weight = T.matrix('imp_weight') if self.imp_weight else None
            comp_dim = 1
        else:
            target = T.tensor3('target')
            imp_weight = T.tensor3('imp_weight') if self.imp_weight else None
            comp_dim = 2

        parameters = ParameterSet()

        self.rnn = neural.Rnn(
            inpt,
            self.n_inpt, self.n_hiddens, self.n_output,
            self.hidden_transfers, self.out_transfer,
            pooling=self.pooling,
            declare=parameters.declare)

        self.loss_layer = simple.SupervisedLoss(
            target, self.rnn.output, loss=self.loss_ident,
            imp_weight=imp_weight,
            declare=parameters.declare,
            comp_dim=comp_dim)

        SupervisedModel.__init__(
            self, inpt=inpt, target=target, output=self.rnn.output,
            loss=self.loss_layer.total,
            parameters=parameters)

        if self.imp_weight:
            self.exprs['imp_weight'] = imp_weight

    def initialize(self,
                   par_std=1, par_std_affine=None, par_std_rec=None,
                   par_std_in=None,
                   sparsify_affine=None, sparsify_rec=None,
                   spectral_radius=None):
        climin.initialize.randomize_normal(self.parameters.data, 0, par_std)
        for i, layer in enumerate(self.rnn.layers):
            if hasattr(layer, 'recurrent'):
                p = self.parameters[layer.recurrent.weights]
                if par_std_rec:
                    climin.initialize.randomize_normal(p, 0, par_std_rec)
                if sparsify_rec:
                    climin.initialize.sparsify_columns(p, sparsify_rec)
                if spectral_radius:
                    climin.initialize.bound_spectral_radius(p, spectral_radius)
                self.parameters[layer.recurrent.initial][...] = 0
            if hasattr(layer, 'affine'):
                p = self.parameters[layer.affine.weights]
                if par_std_affine:
                    if i == 0 and par_std_in:
                        climin.initialize.randomize_normal(p, 0, par_std_in)
                    else:
                        climin.initialize.randomize_normal(p, 0, par_std_affine)
                if sparsify_affine:
                    climin.initialize.sparsify_columns(p, sparsify_affine)

                self.parameters[layer.affine.bias][...] = 0


class SupervisedFastDropoutRnn(BaseRnn, SupervisedModel):

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
        if pooling is not None:
            raise NotImplemented()

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

    def _init_exprs(self):
        inpt = T.tensor3('inpt')
        inpt.tag.test_value = np.zeros((3, 2, self.n_inpt))
        if self.pooling:
            target = T.matrix('target')
            target.tag.test_value = np.zeros((2, self.n_output))
            if self.imp_weight:
                imp_weight = T.matrix('imp_weight')
                imp_weight.tag.test_value = np.ones((2, self.n_output))
            else:
                imp_weight = None
        else:
            target = T.tensor3('target')
            target.tag.test_value = np.zeros((3, 2, self.n_output))
            if self.imp_weight:
                imp_weight = T.tensor3('imp_weight')
                imp_weight.tag.test_value = np.ones((3, 2, self.n_output))
            else:
                imp_weight = None

        parameters = ParameterSet()

        self.rnn = neural.FastDropoutRnn(
            inpt,
            self.n_inpt, self.n_hiddens, self.n_output,
            self.hidden_transfers, self.out_transfer,
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            p_dropout_hidden_to_out=self.p_dropout_hidden_to_out,
            pooling=self.pooling,
            declare=parameters.declare)

        f_loss = lookup(self.loss_ident, vp_loss)
        output = T.concatenate(self.rnn.outputs, 2)
        self.loss_layer = simple.SupervisedLoss(
            target, output, loss=f_loss,
            imp_weight=imp_weight,
            declare=parameters.declare,
            comp_dim=2)

        SupervisedModel.__init__(
            self, inpt=inpt, target=target, output=output,
            loss=self.loss_layer.total,
            parameters=parameters)

        if self.imp_weight:
            self.exprs['imp_weight'] = imp_weight

    def initialize(self,
                   par_std=1, par_std_affine=None, par_std_rec=None,
                   par_std_in=None,
                   sparsify_affine=None, sparsify_rec=None,
                   spectral_radius=None):
        climin.initialize.randomize_normal(self.parameters.data, 0, par_std)
        for i, layer in enumerate(self.rnn.layers):
            if hasattr(layer, 'recurrent'):
                p = self.parameters[layer.recurrent.weights]
                if par_std_rec:
                    climin.initialize.randomize_normal(p, 0, par_std_rec)
                if spectral_radius:
                    climin.initialize.bound_spectral_radius(p, spectral_radius)
                if sparsify_rec:
                    climin.initialize.sparsify_columns(p, sparsify_rec)
                self.parameters[layer.recurrent.initial_mean][...] = 0
                self.parameters[layer.recurrent.initial_std][...] = 1e-8
            if hasattr(layer, 'affine'):
                p = self.parameters[layer.affine.weights]
                if par_std_affine:
                    if i == 0 and par_std_in:
                        climin.initialize.randomize_normal(p, 0, par_std_in)
                    else:
                        climin.initialize.randomize_normal(p, 0, par_std_affine)
                if sparsify_affine:
                    climin.initialize.sparsify_columns(p, sparsify_affine)

                self.parameters[layer.affine.bias][...] = 0
