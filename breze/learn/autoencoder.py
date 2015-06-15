# -*- coding: utf-8 -*-

"""This module has implementations of various auto encoder variants.

We will briefly review auto encoders (AE) in order to establish a common
terminology. The following variants are implemented:

  - Basic Auto Encoder (AE),
  - Denoising Auto Encoder (DAE),
  - Contractive Auto Encoder (CAE),
  - Sparse Auto Encoder (SAE).

The first three are shortly described in Bengio's survey paper [RL]_ while
the latter is covered in Andrew Ng's Unsupervised Feature Learning and Deep
Learning tutorial [UFDLT].

The Higher Order Contractive Auto Encoder and Ranzato's Auto Encoder with
the sparsifying logistic are not implemented.

The auto encoders all follow basic model:

.. math::
   x' = s'(s(xW + b)W' + b')

where a loss :math:`L(x, x')` is minimized which encourages the auto encoder
to reconstruct the input. Examples of a loss are the _mean of squares_ loss or
the _cross entropy_ loss.

Here, :math:`x` is the input to the model, :math:`W` and :math:`W'` are
weight matrices, :math:`b` and :math:`b'` biases, :math:`s` and :math:`s'`
element wise non-linearities. :math:`x'` is what we call the reconstruction or
reconstructed input.

The idea, as can be read in the references, is that the feature/hidden/latent
representation

.. math::
   h = s(xW + b)

is somewhat meaningful.

Often, :math:`W' = W^T` is explicitly enforced to reduce the number of
parameters. This can be specified by setting `tied_weights` to True during
creation of the respective objective and also is the default.

Furthermore, the parameters of the model in the parameter set are given as
:math:`\lbrace W, W', b, b' \\rbrace`. The respective fields point (accessed
via `.`) as attributes in the `parameter` field of the auto encoder object to
the respective Theano variables. As keys (accessed via `[]`) they point to the
respective numpy arrays.

========== =============== =====================================================
Notation    Field           Note
========== =============== =====================================================
:math:`W`  `in_to_hidden`
:math:`W'` `hidden_to_out` points to `in_to_hidden.T` if `tied_weights` is set.
:math:`b`  `hidden_bias`
:math:`b'` `out_bias`
========== =============== =====================================================

The other parts of the formula, :math:`s`, :math:`s'`, :math:`L` are set as
arguments to the constructor. For an exact way of doing so see
:ref:`specifying-functions`. The following table gives which argument to the
constructor belongs to which symbol in the above formulas.

========== ====================
Notation    Argument
========== ====================
:math:`s`  ``hidden_transfer``
:math:`s'` ``out_transfer``
:math:`L`  ``loss``
========== ====================

The special models all have a additional losses and parameters, which will are
described in the corresponding paragraphs.

.. [RL] http://arxiv.org/abs/1206.5538
.. [UFDLT] http://ufldl.stanford.edu/
"""

import numpy as np
import theano
import theano.tensor as T

from breze.arch.component import loss as loss_
from breze.arch.util import ParameterSet, lookup, get_named_variables
from breze.arch.component import corrupt
from breze.arch.component.common import supervised_loss

from breze.learn.base import (
    UnsupervisedModel, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)

from breze.arch.construct import neural
from breze.learn.utils import theano_floatx

theano.config.compute_test_value = 'raise'


class AutoEncoder(UnsupervisedModel,
                  TransformBrezeWrapperMixin, ReconstructBrezeWrapperMixin):
    """Auto Encoder class.

    Intended as a base class for all other auto encoders.

    Attributes
    ----------

    batch_size : integer
        Numer of samples to look at at each gradient update.

    tied_weights : boolean
        Indicates whether the decoding matrix is the transpose of the encoding
        matrix.

    exprs : dictionary
        Dictionary containing the different symbolic variables of the model.

    hidden_transfers: list of string or functions
        Transfer function being used.

    n_hiddens : list of integer
        Number of features/hidden units to detect per layer.
    """

    transform_expr_name = 'feature'

    def __init__(self, n_inpt, n_hiddens, hidden_transfers,
                 out_transfer='identity', loss_ident='squared', tied_weights=True,
                 code_idx=None,
                 batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create an AutoEncoder object.

        Parameters
        ----------

        n_inpt : integer
            Input dimensionality of the data. (And thus, also output dimensionality.)

        n_hiddens : list of integer
            Dimensionality of the hidden feature dimension per layer.

        hidden_transfers : list of strings or functions
            List of transfer functions to use for the hidden units. Each item
            can be a string referring any function found in
            ``breze.arch.component.transfer`` or a function that given an
            ``(n, d)`` array returns an ``(n, d)`` array as theano expressions.

        out_transfer : string or function
            Output transfer function of the linear auto encoder for calculation
            of the reconstruction cost.

        loss : string or function
            Loss which is going to be optimized. This can either be a string
            and reference a loss function found in ``breze.component.loss`` or
            a function which takes two theano tensors (one being the target the
            other the output of the network) and returns a theano scalar.

        tied_weights : boolean, optional [default: True]
            Flag indicating whether to use tied weights.

        batch_size : integer
            Number of examples per batch when calculing the loss and its
            derivatives. None means to use all samples every time.

        optimizer: string or pair
            Can be either a string or a pair. In any case,
            ``climin.util.optimizer`` is used to construct an optimizer. In the
            case of a string, the string is used as an identifier for the
            optimizer which is then instantiated with default arguments. If a
            pair, expected to be (`identifier`, `kwargs`) for more fine control
            of the optimizer.

        max_iter : integer
            Maximum number of optimization iterations to perform.

        verbose : boolean
            Flag indicating whether to print out information during fitting.
        """
        self.n_inpt = n_inpt
        self.n_hiddens = n_hiddens
        self.hidden_transfers = hidden_transfers
        self.out_transfer = out_transfer
        self.loss_ident = loss_ident
        self.tied_weights = tied_weights
        self.code_idx = code_idx if code_idx is not None else len(n_hiddens) / 2

        self.batch_size = batch_size
        self.optimizer = optimizer

        # TODO move this somehwere central (Mixins)
        self.f_transform = None
        self.f_reconstruct = None

        self.max_iter = max_iter
        self.verbose = verbose

        self._init_exprs()


    def _init_exprs(self):
        inpt = T.matrix('inpt')
        inpt.tag.test_value, = theano_floatx(np.ones((3, self.n_inpt)))

        parameters = ParameterSet()

        self.mlp = neural.Mlp(
            inpt, self.n_inpt,
            self.n_hiddens,
            self.n_inpt,
            self.hidden_transfers, self.out_transfer,
            declare=parameters.declare)

        output = self.mlp.output

        n_dim = inpt.ndim # to be used in the arguments of supervised_loss
        # to define the coord_axis in case of inpt_dim > 2
        rec_loss_coord = supervised_loss(
            inpt, output, self.loss_ident, coord_axis=1)['loss_coord_wise']
        rec_loss_sample_wise = rec_loss_coord.sum(axis=1)
        rec_loss = rec_loss_sample_wise.mean()


        UnsupervisedModel.__init__(self, inpt=inpt,
                                 output=output,
                                 loss=rec_loss,
                                 parameters=parameters)

        self.feature = self.mlp.layers[self.code_idx].output
        self.filters_in_to_hidden = parameters[self.mlp.layers[0].weights]
        self.filters_hidden_to_out = parameters[self.mlp.layers[-1].weights]


class SparseAutoEncoder(AutoEncoder):
    """Implementation of the Sparse Auto Encoder (SAE) as described in [UFDLT]_.

    The SAE discourages trivial solutions to the reconstruction loss by
    adding a penalty term to the loss which encourages sparse activations of the
    hidden units. This is employed by specifying a `sparsity target`, which is
    the desired average activation of the hidden units. Additionally, a specific
    distance measure is defined, the `sparsity loss`, to quantify the divergence
    from this desired activity. That is then added to the reconstruction loss:

    .. math::
       L_{sae} = L(x, x') + \lambda d(\hat{h}, t).

    Here, :math:`\hat{h} = \\frac{1}{NH}\sum_{i, j} h_{ij}` where :math:`h_{ij}`
    is given as the `j`'th hidden unit in the `i`'th training sample
    with `N` training samples and `H` hidden units in total. The desired
    activation is given by `t`.

    We now give a table of the corresponding argument names in the
    SparseAutoEncoder initializer. The fields of the resulting objects are
    named the same.

    =============== ===================
    Notation        Argument/Field name
    =============== ===================
    :math:`\lambda` ``c_sparsity``
    :math:`d`       ``sparsity_loss``
    :math:`t`       ``sparsity_target``
    =============== ===================

    A typical choice for the sparsity loss is the KL divergence between two
    Bernoulli variables, which can be specified by ``bern_bern_kl``. Typical
    values for the sparsity target are between 0.01 and 0.05.

    The part of the loss which corresponds to the regularization term is
    identified by ``sparsity_loss``. The reconstruction part is identified
    by ``reconstruct_loss``.

    .. [UFDLT] http://ufldl.stanford.edu/


    Attributes
    ----------

    Same attributes as ``AutoEncoder`` objects.

    c_sparsity : float
        Coefficient weighing the sparsity cost in comparison to the
        reconstruction cost.

    sparsity_loss : float
        Sparsity part of the loss which is going to be optimized. This can
        either be a string and reference a loss function found in
        ``breze.component.loss`` or a function which takes two theano tensors
        (one being the output of the network, the other some target) and
        returns a theano scalar.

    sparsity_target : float
        Subtract this value from each hidden unit before applying the sparsity
        loss.
    """

    def __init__(self, n_inpt, n_hiddens, hidden_transfers,
                 out_transfer='identity', loss='squared',
                 c_sparsity=1, sparsity_loss='bern_bern_kl',
                 sparsity_target=0.01,
                 tied_weights=True, code_idx=None, batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create a SparseAutoEncoder object.

        Parameters
        ----------

        All parameters from the ``AutoEncoder`` apply as well.

        c_sparsity : float
            Coefficient weighing the sparsity cost in comparison to the
            reconstruction cost.

        sparsity_loss : float
            Sparsity part of the loss which is going to be optimized. This can
            either be a string and reference a loss function found in
            ``breze.component.distance`` or a function which takes two theano
            tensors (one being the output of the network, the other some
            target) and returns a theano scalar.

        sparsity_target : float
            Subtract this value from each hidden unit before applying the
            sparsity loss.
        """
        self.c_sparsity = c_sparsity
        self.sparsity_loss = sparsity_loss
        self.sparsity_target = sparsity_target
        super(SparseAutoEncoder, self).__init__(
            n_inpt, n_hiddens, hidden_transfers, out_transfer,
            loss,
            tied_weights=tied_weights,
            code_idx=code_idx,
            batch_size=batch_size,
            optimizer=optimizer,
            max_iter=max_iter,
            verbose=verbose)

    def _init_exprs(self):
        super(SparseAutoEncoder, self)._init_exprs()
        f_sparsity_loss = lookup(self.sparsity_loss, loss_)
        sparsity_loss = f_sparsity_loss(
            self.sparsity_target, self.feature.mean(axis=0)).sum()
        loss = self.loss + self.c_sparsity * sparsity_loss

        self.exprs.update(get_named_variables(locals(), overwrite=True))


class ContractiveAutoEncoder(AutoEncoder):
    """Implementation of the Contractive Auto Encoder (CAE) as described in
    [CAE]_.

    The CAE discourages trivial solutions to the reconstruction loss by
    adding a penalty term to the loss which encourages the Jacobian of the
    activations of the hidden units to be flat. This is employed by adding the
    Frobenious norm of the Jacobians around the training point as a regularizer.

    .. math::
       L_{cae} = L(x, x') +
       \lambda \sum_{ij}(\\frac{\partial{h_i}}{\partial{x_j}})^2

    :math:`\lambda` can be specified via the ``c_jacobian`` argument and is
    availabe as a field of the object.

    The part of the loss which corresponds to the regularization term is
    identified by ``jacobian_loss``. The reconstruction part is identified
    by ``reconstruct_loss``.

    .. [CAE] Contractive auto-encoders: Explicit invariance during
       feature extraction, Rifai et al (2011)

    Attributes
    ----------

    All attributes of ``AutoEncoder`` objects apply as well.

    c_jacobian : float
        Coefficient weighing the Jacobian cost in comparison to the
        reconstruction cost.
    """

    def __init__(self, n_inpt, n_hiddens, hidden_transfers,
                 out_transfer='identity', loss='squared',
                 c_jacobian=1,
                 tied_weights=True, code_idx=None, batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create a ContractiveAutoEncoder object.

        Parameters
        ----------

        All parameters from the ``AutoEncoder`` class apply as well.

        c_jacobian : float
            Coefficient weighing the Jacobian cost in comparison to the
            reconstruction cost.
        """
        self.c_jacobian = c_jacobian
        super(ContractiveAutoEncoder, self).__init__(
            n_inpt, n_hiddens, hidden_transfers, out_transfer,
            loss,
            tied_weights=tied_weights,
            code_idx=code_idx,
            batch_size=batch_size,
            optimizer=optimizer,
            max_iter=max_iter,
            verbose=verbose)

    def _init_exprs(self):
        super(ContractiveAutoEncoder, self)._init_exprs()
        jacobian_loss = T.sum(T.grad(self.feature.mean(axis=0).sum(),
                              self.inpt).mean(axis=0) ** 2)
        loss = self.loss + self.c_jacobian * jacobian_loss

        self.exprs.update(get_named_variables(locals(), overwrite=True))


class DenoisingAutoEncoder(AutoEncoder):
    """Implementation of the Denoising Auto Encoder (DAE) as described in
    [DAE]_.

    The DAE discourages trivial solutions to the reconstruction loss by
    corrupting the input data with noise, while still trying to recover the
    uncorrupted data from that. Formally, this can be written as

    .. math::
       L_{dae} = \mathbb{E}_{\hat{x} \sim q(\hat{x}|x)} L(\hat{x}, x')

    where :math:`q` is a corruption distribution. In this implementation,
    the corruption distribution can be either additive Gaussian noise or
    blink noise, which sets an input component with a certain probability
    to zero. In the former case, the distribution can be specified by
    setting the construction argument ``noise_type`` to ``gauss``; in the latter
    case, it is to be set to ``blink``. In both cases, the noise parameter
    is specified via ``c_noise``. In the Gaussian case, this refers to the
    standard deviation of the distribution. In the blink case, it is the
    probability of setting an input component to 0.

    .. [DAE] Extracting and Composing Robust Features with Denoising
       Autoencoders, Vincent et al (2008).
    Attributes
    ----------

    All attributes of ``AutoEncoder`` objects apply as well.

    noise_type : {'blink', 'gauss'}
        Specifies the type of noise used, either 'gauss' or 'blink'. The former
        adds Gaussian noise, the latter sets inputs random to zero.

    c_noise : float
        Standard deviation of the noise in case of Gaussian noise, "set to
        zero" probability in case of blink noise.

    """

    def __init__(self, n_inpt, n_hiddens, hidden_transfers='sigmoid',
                 out_transfer='identity', loss='squared',
                 noise_type='gauss', c_noise=.2,
                 tied_weights=True,
                 code_idx=None,
                 batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create a DenoisingAutoEncoder object.

        Parameters
        ----------

        All parameters from the ``AutoEncoder`` class apply as well.

        noise_type : {'blink', 'gauss'}
            Specifies the type of noise used, either 'gauss' or
            'blink'. The former adds Gaussian noise, the latter sets inputs
            random to zero.

        c_noise : float
            Standard deviation of the noise in case of Gaussian
            noise, "set to zero" probability in case of blink noise.
        """
        self.c_noise = c_noise
        self.noise_type = noise_type

        super(DenoisingAutoEncoder, self).__init__(
            n_inpt, n_hiddens, hidden_transfers, out_transfer,
            loss,
            tied_weights=tied_weights,
            code_idx=code_idx,
            batch_size=batch_size,
            optimizer=optimizer,
            max_iter=max_iter,
            verbose=verbose)

    def _init_exprs(self):
        # Here we need to replace the input with a corrupted version. If we do
        # so naively by calling clone on the loss, the targets (which are
        # identical to the inputs in thesense of identity in programming) the
        # targets will be replaced as well. Instead, we just want to thave the
        # inputs replaced. Thus we first clone the output of the model and
        # replace the input with the corrupted input. This will not change the
        # targets. Afterwards, we put that corruption into the loss as well.
        super(DenoisingAutoEncoder, self)._init_exprs()
        if self.noise_type == 'gauss':
            corrupted_inpt = corrupt.gaussian_perturb(
                self.inpt, self.c_noise)
        elif self.noise_type == 'mask':
            corrupted_inpt = corrupt.mask(
                self.inpt, self.c_noise)

        output_from_corrupt = theano.clone(
            self.output,
            {self.inpt: corrupted_inpt}
        )

        score = self.loss
        loss = theano.clone(
            self.loss,
            {self.output: output_from_corrupt})

        self.exprs.update(get_named_variables(locals(), overwrite=True))
