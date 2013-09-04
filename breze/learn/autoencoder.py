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

from breze.arch.model.feature import (
    AutoEncoder as _AutoEncoder,
    SparseAutoEncoder as _SparseAutoEncoder,
    ContractiveAutoEncoder as _ContractiveAutoEncoder,
    DenoisingAutoEncoder as _DenoisingAutoEncoder)
from breze.learn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)


class AutoEncoder(_AutoEncoder, UnsupervisedBrezeWrapperBase,
                  TransformBrezeWrapperMixin, ReconstructBrezeWrapperMixin):
    """The basic Auto Encoder. What is written in the module
    documentation applies to this class."""

    transform_expr_name = 'hidden'

    def __init__(self, n_inpt, n_hidden, hidden_transfer='identity',
                 out_transfer='identity', loss='squared', tied_weights=True,
                 batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create an AutoEncoder object.

        :param n_inpt: Input dimensionality of the data.
        :param n_hidden: Dimensionality of the hidden feature dimension.
        :param hidden_transfer: Transfer function to use for the hidden units.
            Can be a string referring any function found in
            ``breze.component.transfer`` or a function that given an (n, d)
            array returns an (n, d) array as theano expressions.
        :param out_transfer: Output transfer function of the linear auto encoder
            for calculation of the reconstruction cost.
        :param loss: Loss which is going to be optimized. This can either be a
            string and reference a loss function found in
            ``breze.component.distance`` or a function which takes two theano
            tensors (one being the output of the network, the other some target)
            and returns a theano scalar.
        :param tied_weights: Flag indicating whether to use tied weights.
        :param batch_size: Number of examples per batch when calculing the loss
            and its derivatives. None means to use all samples every time.
        :param optimizer: Can be either a string or a pair. In any case,
            climin.util.optimizer is used to construct an optimizer. In the case
            of a string, the string is used as an identifier for the optimizer
            which is then instantiated with default arguments. If a pair,
            expected to be (`identifier`, `kwargs`) for more fine control of the
            optimizer.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(AutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer, loss, tied_weights)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.f_transform = None
        self.f_reconstruct = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)
        self.max_iter = max_iter
        self.verbose = verbose


class SparseAutoEncoder(_SparseAutoEncoder, UnsupervisedBrezeWrapperBase,
                        TransformBrezeWrapperMixin,
                        ReconstructBrezeWrapperMixin):
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
    """

    transform_expr_name = 'hidden'

    def __init__(self, n_inpt, n_hidden, hidden_transfer='sigmoid',
                 out_transfer='identity', reconstruct_loss='squared',
                 c_sparsity=1, sparsity_loss='bern_bern_kl',
                 sparsity_target=0.01,
                 tied_weights=True, batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create a SparseAutoEncoder object.

        :param n_inpt: Input dimensionality of the data.
        :param n_hidden: Dimensionality of the hidden feature dimension.
        :param hidden_transfer: Transfer function to use for the hidden units.
            Can be a string referring any function found in
            ``breze.component.transfer`` or a function that given an (n, d)
            array returns an (n, d) array as theano expressions.
        :param out_transfer: Output transfer function of the linear auto encoder
            for calculation of the reconstruction cost.
        :param reconstruct_loss: Reconstruction part of the loss which is
            going to be optimized. This can either be a string and reference a
            loss function found in ``breze.component.distance`` or a function
            which takes two theano tensors (one being the output of the network,
            the other some target) and returns a theano scalar.
        :param c_sparsity: Coefficient weighing the sparsity cost in comparison
            to the reconstruction cost.
        :param sparsity_loss: Sparsity part of the loss which is
            going to be optimized. This can either be a string and reference a
            loss function found in ``breze.component.distance`` or a function
            which takes two theano tensors (one being the output of the network,
            the other some target) and returns a theano scalar.
        :param sparsity_target: Subtract this value from each hidden unit before
            applying the sparsity loss.
        :param tied_weights: Flag indicating whether to use tied weights.
        :param batch_size: Number of examples per batch when calculing the loss
            and its derivatives. None means to use all samples every time.
        :param optimizer: Can be either a string or a pair. In any case,
            climin.util.optimizer is used to construct an optimizer. In the case
            of a string, the string is used as an identifier for the optimizer
            which is then instantiated with default arguments. If a pair,
            expected to be (`identifier`, `kwargs`) for more fine control of the
            optimizer.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(SparseAutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer,
            reconstruct_loss, c_sparsity, sparsity_loss, sparsity_target,
            tied_weights)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.f_transform = None
        self.f_reconstruct = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)
        self.max_iter = max_iter
        self.verbose = verbose


class ContractiveAutoEncoder(_ContractiveAutoEncoder,
                             UnsupervisedBrezeWrapperBase,
                             TransformBrezeWrapperMixin,
                             ReconstructBrezeWrapperMixin):
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
    """

    transform_expr_name = 'hidden'

    def __init__(self, n_inpt, n_hidden, hidden_transfer='sigmoid',
                 out_transfer='identity', reconstruct_loss='squared',
                 c_jacobian=1, tied_weights=True, batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create a ContractiveAutoEncoder object.

        :param n_inpt: Input dimensionality of the data.
        :param n_hidden: Dimensionality of the hidden feature dimension.
        :param hidden_transfer: Transfer function to use for the hidden units.
            Can be a string referring any function found in
            ``breze.component.transfer`` or a function that given an (n, d)
            array returns an (n, d) array as theano expressions.
        :param out_transfer: Output transfer function of the linear auto encoder
            for calculation of the reconstruction cost.
        :param reconstruct_loss: Reconstruction part of the loss which is
            going to be optimized. This can either be a string and reference a
            loss function found in ``breze.component.distance`` or a function
            which takes two theano tensors (one being the output of the network,
            the other some target) and returns a theano scalar.
        :param c_jacobian: Coefficient weighing the Jacobian cost in comparison
            to the reconstruction cost.
        :param tied_weights: Flag indicating whether to use tied weights.
        :param batch_size: Number of examples per batch when calculing the loss
            and its derivatives. None means to use all samples every time.
        :param optimizer: Can be either a string or a pair. In any case,
            climin.util.optimizer is used to construct an optimizer. In the case
            of a string, the string is used as an identifier for the optimizer
            which is then instantiated with default arguments. If a pair,
            expected to be (`identifier`, `kwargs`) for more fine control of the
            optimizer.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(ContractiveAutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer,
            reconstruct_loss, c_jacobian,
            tied_weights)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.f_transform = None
        self.f_reconstruct = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape).astype(theano.config.floatX)
        self.max_iter = max_iter
        self.verbose = verbose


class DenoisingAutoEncoder(_DenoisingAutoEncoder, UnsupervisedBrezeWrapperBase,
                           TransformBrezeWrapperMixin,
                           ReconstructBrezeWrapperMixin):
    """Implementation of the Denoising Auto Encoder (DAE) as described in
    [DAE]_.

    The SAE discourages trivial solutions to the reconstruction loss by
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
    """
    transform_expr_name = 'hidden'

    def __init__(self, n_inpt, n_hidden, hidden_transfer='sigmoid',
                 out_transfer='identity', reconstruct_loss='squared',
                 noise_type='gauss', c_noise=.2,
                 tied_weights=True, batch_size=None,
                 optimizer='lbfgs', max_iter=1000, verbose=False):
        """Create a DenoisingAutoEncoder object.

        :param n_inpt: Input dimensionality of the data.
        :param n_hidden: Dimensionality of the hidden feature dimension.
        :param hidden_transfer: Transfer function to use for the hidden units.
            Can be a string referring any function found in
            ``breze.component.transfer`` or a function that given an (n, d)
            array returns an (n, d) array as theano expressions.
        :param out_transfer: Output transfer function of the linear auto encoder
            for calculation of the reconstruction cost.
        :param reconstruct_loss: Reconstruction part of the loss which is
            going to be optimized. This can either be a string and reference a
            loss function found in ``breze.component.distance`` or a function
            which takes two theano tensors (one being the output of the network,
            the other some target) and returns a theano scalar.
        :param noise_type: Specifies the type of noise used, either 'gauss' or
            'blink'. The former adds Gaussian noise, the latter sets inputs
            random to zero.
        :param c_noise: Standard deviation of the noise in case of Gaussian
            noise, "set to zero" probability in case of blink noise.
        :param tied_weights: Flag indicating whether to use tied weights.
        :param batch_size: Number of examples per batch when calculing the loss
            and its derivatives. None means to use all samples every time.
        :param optimizer: Can be either a string or a pair. In any case,
            climin.util.optimizer is used to construct an optimizer. In the case
            of a string, the string is used as an identifier for the optimizer
            which is then instantiated with default arguments. If a pair,
            expected to be (`identifier`, `kwargs`) for more fine control of the
            optimizer.
        :param max_iter: Maximum number of optimization iterations to perform.
        :param verbose: Flag indicating whether to print out information during
            fitting.
        """
        super(DenoisingAutoEncoder, self).__init__(
            n_inpt, n_hidden, hidden_transfer, out_transfer,
            reconstruct_loss, noise_type, c_noise,
            tied_weights)
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.f_transform = None
        self.f_reconstruct = None
        self.parameters.data[:] = np.random.standard_normal(
            self.parameters.data.shape, dtype=theano.config.floatX)
        self.max_iter = max_iter
        self.verbose = verbose
