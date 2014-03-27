# -*- coding: utf-8 -*-

"""Module for learning various types of multilayer perceptrons."""


import itertools

import climin
import climin.util
import climin.gd
from climin.project import max_length_columns

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from breze.arch.model.neural import mlp
from breze.arch.model.varprop import mlp as varprop_mlp
from breze.arch.component.varprop.common import supervised_loss as varprop_supervised_loss
from breze.arch.component.common import supervised_loss
from breze.learn.base import SupervisedBrezeWrapperBase
from breze.arch.util import ParameterSet, Model


# TODO Mlp docs are loss missing


class Mlp(Model, SupervisedBrezeWrapperBase):
    """Multilayer perceptron class.

    This implementation uses a stack of affine mappings with a subsequent
    non linearity each.

    Parameters
    ----------

    n_inpt : integer
        Dimensionality of a single input.

    n_hiddens : list of integers
        List of ``k`` integers, where ``k`` is thenumber of layers. Each gives
        the size of the corresponding layer.

    n_output : integer
        Dimensionality of a single output.

    hidden_transfers : list, each item either string or function
        Transfer functions for each of the layers. Can be either a string which
        is then used to look up a transfer function in
        ``breze.component.transfer`` or a function that given a Theano tensor
        returns a tensor of the same shape.

    out_transfer : string or function
        Either a string to look up a function in ``breze.component.transfer``
        or a function that given a Theano tensor returns a tensor of the same
        shape.

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

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 optimizer='lbfgs',
                 batch_size=None,
                 max_iter=1000, verbose=False):
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.loss = loss

        self.optimizer = optimizer
        self.batch_size = batch_size

        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None

        super(Mlp, self).__init__()

    def _init_pars(self):
        spec = mlp.parameters(self.n_inpt, self.n_hiddens, self.n_output)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt': T.matrix('inpt'),
            'target': T.matrix('target')
        }
        P = self.parameters

        n_layers = len(self.n_hiddens)
        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(n_layers - 1)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(n_layers)]

        self.exprs.update(mlp.exprs(
            self.exprs['inpt'],
            P.in_to_hidden, hidden_to_hiddens, P.hidden_to_out,
            hidden_biases, P.out_bias,
            self.hidden_transfers, self.out_transfer))

        self.exprs.update(supervised_loss(
            self.exprs['target'], self.exprs['output'], self.loss))


def dropout_optimizer_conf(
        steprate_0=1, steprate_decay=0.998, momentum_0=0.5,
        momentum_eq=0.99, n_momentum_anneal_steps=500,
        n_repeats=500):
    """Return a dictionary suitable for climin.util.optimizer which
    specifies the standard optimizer for dropout mlps."""
    steprate = climin.gd.decaying(steprate_0, steprate_decay)
    momentum = climin.gd.linear_annealing(
        momentum_0, momentum_eq, n_momentum_anneal_steps)

    # Define another time for steprate calculcation.
    momentum2 = climin.gd.linear_annealing(
        momentum_0, momentum_eq, n_momentum_anneal_steps)
    steprate = ((1 - j) * i for i, j in itertools.izip(steprate, momentum2))

    steprate = climin.gd.repeater(steprate, n_repeats)
    momentum = climin.gd.repeater(momentum, n_repeats)

    return 'gd', {
        'steprate': steprate,
        'momentum': momentum,
    }


class DropoutMlp(Mlp):
    """Class representing an MLP that is trained with dropout [D]_.

    The gist of this method is that hidden units and input units are "zerod out"
    with a certain probability.

    References
    ----------
    .. [D] Hinton, Geoffrey E., et al.
           "Improving neural networks by preventing co-adaptation of feature
           detectors." arXiv preprint arXiv:1207.0580 (2012).


    Attributes
    ----------

    Same attributes as an ``Mlp`` object.

    p_dropout_inpt : float
        Probability that an input unit is ommitted during a pass.

    p_dropout_hidden : float
        Probability that an input unit is ommitted during a pass.

    max_length : float
        Maximum squared length of a weight vector into a unit. After each
        update, the weight vectors will projected to be shorter.
    """

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 p_dropout_inpt=.2, p_dropout_hiddens=.5,
                 max_length=None,
                 optimizer='rprop',
                 batch_size=None,
                 max_iter=1000, verbose=False):
        """Create a DropoutMlp object.


        Parameters
        ----------

        Same attributes as an ``Mlp`` object.

        p_dropout_inpt : float
            Probability that an input unit is ommitted during a pass.

        p_dropout_hiddens : list of floats
            List of which each item gives the probability that a hidden unit
            of that layer is omitted during a pass.

        """
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens
        super(DropoutMlp, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer, loss,
            optimizer=optimizer, batch_size=batch_size, max_iter=max_iter,
            verbose=verbose)


class FastDropoutNetwork(Model, SupervisedBrezeWrapperBase):
    """Class representing an MLP that is trained with fast dropout [FD]_.

    This method employs a smooth approximation of dropout training.


    References
    ----------
    .. [FD] Wang, Sida, and Christopher Manning.
            "Fast dropout training."
            Proceedings of the 30th International Conference on Machine
            Learning (ICML-13). 2013.


    Attributes
    ----------

    Same attributes as an ``Mlp`` object.

    p_dropout_inpt : float
        Probability that an input unit is ommitted during a pass.

    p_dropout_hiddens : list of floats
        Each item constitues the probability that a hidden unit of the
        corresponding layer is ommitted during a pass.

    inpt_var : float
        Assumed variance of the inputs. "quasi zero" per default.
    """

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 optimizer='lbfgs',
                 batch_size=None,
                 p_dropout_inpt=.2,
                 p_dropout_hiddens=.5,
                 max_length=None,
                 inpt_var=1e-8,
                 max_iter=1000, verbose=False):
        """Create a FastDropoutMlp object.


        Parameters
        ----------

        Same parameters as an ``Mlp`` object.

        p_dropout_inpt : float
            Probability that an input unit is ommitted during a pass.

        p_dropout_hidden : float
            Probability that an input unit is ommitted during a pass.

        max_length : float or None
            Maximum squared length of a weight vector into a unit. After each
            update, the weight vectors will projected to be shorter.
            If None, no projection is performed.
        """
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.loss = loss

        self.p_dropout_inpt = p_dropout_inpt
        if isinstance(p_dropout_hiddens, float):
            self.p_dropout_hiddens = [p_dropout_hiddens]
        else:
            self.p_dropout_hiddens = p_dropout_hiddens

        if not all(0 < i < 1 for i in [p_dropout_inpt] + self.p_dropout_hiddens):
            raise ValueError('dropout rates have to be in (0, 1)')

        self.max_length = max_length
        self.inpt_var = inpt_var

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose

        super(FastDropoutNetwork, self).__init__()

    def _init_pars(self):
        spec = varprop_mlp.parameters(
            self.n_inpt, self.n_hiddens, self.n_output, False)
        self.parameters = ParameterSet(**spec)
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)

    def _init_exprs(self):
        self.exprs = {
            'inpt_mean': T.matrix('inpt_mean'),
            'target': T.matrix('target')}
        P = self.parameters

        hidden_to_hiddens = [getattr(P, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(P, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        inpt_var = T.zeros_like(self.exprs['inpt_mean']) + self.inpt_var

        self.exprs.update(varprop_mlp.exprs(
            self.exprs['inpt_mean'], inpt_var, self.exprs['target'],
            P.in_to_hidden,
            hidden_to_hiddens,
            P.hidden_to_out,
            hidden_biases,
            [1 for _ in hidden_biases],
            P.out_bias,
            1,
            self.hidden_transfers, self.out_transfer,
            self.p_dropout_inpt, self.p_dropout_hiddens))

        self.exprs['inpt'] = self.exprs['inpt_mean']

        self.exprs.update(varprop_supervised_loss(
            self.exprs['target'], self.exprs['output'], self.loss))

    def iter_fit(self, X, Z, info_opt=None):
        """Iteratively fit the parameters of the model to the given data with
        the given error function.

        Each iteration of the learning algorithm is an iteration of the returned
        iterator. The model is in a valid state after each iteration, so that
        the optimization can be broken any time by the caller.

        This method does `not` respect the max_iter attribute.

        Parameters
        ----------

        X : array_like
            Input data. 2D array of the shape ``(n ,d)`` where ``n`` is the
            number of data samples and ``d`` is the dimensionality of a single
            data sample.
        Z : array_like
            Target data. 2D array of the shape ``(n, l)`` array where ``n`` is
            defined as in ``X``, but ``l`` is the dimensionality of a single
            output.
        """
        for info in super(FastDropoutNetwork, self).iter_fit(X, Z, info_opt=info_opt):
            yield info
            if self.max_length is not None:
                W = self.parameters['in_to_hidden']
                max_length_columns(W, self.max_length)

                n_layers = len(self.n_hiddens)
                for i in range(n_layers - 1):
                    W = self.parameters['hidden_to_hidden_%i' % i]
                    max_length_columns(W, self.max_length)
                W = self.parameters['hidden_to_out']
                max_length_columns(W, self.max_length)
