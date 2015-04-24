# -*- coding: utf-8 -*-

"""Module for learning various types of multilayer perceptrons."""


import itertools

import climin
import climin.util
import climin.gd

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from breze.arch.component.varprop import loss as loss_
from breze.arch.util import lookup
from breze.arch.construct.base import SupervisedStack
from breze.arch.construct.layer import simple, varprop


# TODO Mlp docs are loss missing


class Mlp(SupervisedStack):
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

    imp_weight : boolean
        Flag indicating whether importance weights are used.

    max_iter : int
        Maximum number of optimization iterations to perform. Only respected
        during``.fit()``, not ``.iter_fit()``.

    verbose : boolean
        Flag indicating whether to print out information during fitting.
    """

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer, loss,
                 imp_weight=False,
                 optimizer='lbfgs',
                 batch_size=None,
                 max_iter=1000, verbose=False):
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.n_output = n_output
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self._loss = loss

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.imp_weight = imp_weight

        self.max_iter = max_iter
        self.verbose = verbose

        self.f_predict = None

        self._init_layers()

    def _init_layers(self):
        inpt = T.matrix('inpt')
        target = T.matrix('target')
        imp_weight = T.matrix('imp_weight') if self.imp_weight else None

        n_incoming = [self.n_inpt] + self.n_hiddens
        n_outgoing = self.n_hiddens + [self.n_output]
        transfers = self.hidden_transfers + [self.out_transfer]

        layers = [simple.AffineNonlinear(n, m, f) for n, m, f
                  in zip(n_incoming, n_outgoing, transfers)]
        loss = simple.SupervisedLoss(self._loss, target, imp_weight=imp_weight)

        super(Mlp, self).__init__(layers, loss)

        inpt.tag.test_value = np.zeros(
            (10, self.n_inpt)).astype(theano.config.floatX)
        target.tag.test_value = np.zeros(
            (10, self.n_output)).astype(theano.config.floatX)
        if imp_weight is not None:
            imp_weight.tag.test_value = np.zeros(
                (10, self.n_output)).astype(theano.config.floatX)

        self.forward(inpt)


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


class FastDropoutNetwork(Mlp):
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
                 imp_weight=False,
                 optimizer='lbfgs',
                 batch_size=None,
                 p_dropout_inpt=.2,
                 p_dropout_hiddens=.5,
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
        self.p_dropout_inpt = p_dropout_inpt
        if isinstance(p_dropout_hiddens, float):
            self.p_dropout_hiddens = [p_dropout_hiddens] * len(n_hiddens)
        else:
            self.p_dropout_hiddens = p_dropout_hiddens

        p_dropouts = [p_dropout_inpt] + self.p_dropout_hiddens
        if not all(0 < i < 1 for i in p_dropouts):
            raise ValueError('dropout rates have to be in (0, 1)')

        self.inpt_var = inpt_var

        super(FastDropoutNetwork, self).__init__(
            n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer, loss,
            imp_weight, optimizer, batch_size, max_iter, verbose)

    def _init_layers(self):
        target = T.matrix('target')
        imp_weight = T.matrix('imp_weight') if self.imp_weight else None

        n_incoming = [self.n_inpt] + self.n_hiddens
        n_outgoing = self.n_hiddens + [self.n_output]
        transfers = self.hidden_transfers + [self.out_transfer]
        p_dropouts = [self.p_dropout_inpt] + self.p_dropout_hiddens

        layers = [varprop.AugmentVariance(self.inpt_var)]
        for n, m, f, d in zip(n_incoming, n_outgoing, transfers, p_dropouts):
            layers.append(varprop.FastDropout(d))
            layers.append(varprop.AffineNonlinear(n, m, f))
        layers.append(simple.Concatenate())

        f_loss = lookup(self._loss, loss_)
        loss = simple.SupervisedLoss(f_loss, target, imp_weight=imp_weight)

        super(Mlp, self).__init__(layers, loss)
        inpt = T.matrix('inpt')

        inpt.tag.test_value = np.zeros(
            (10, self.n_inpt)).astype(theano.config.floatX)
        target.tag.test_value = np.zeros(
            (10, self.n_output)).astype(theano.config.floatX)
        if imp_weight is not None:
            imp_weight.tag.test_value = np.zeros(
                (10, self.n_output)).astype(theano.config.floatX)

        self.forward(inpt)
