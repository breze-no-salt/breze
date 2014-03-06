# -*- coding: utf-8 -*-


import theano.tensor as T

from ...util import get_named_variables


# TODO docstring examples


def parameters(n_feature, n_inpt):
    """Return the parameter specification dictionary for a sparse coding
    model.

    Parameters
    ----------

    n_inpt : integer
        Number of inpus to the model.

    n_output : integer
        Number of outputs of the model.

    Returns
    -------

    res : dict
        Specification of the parameters.
    """"

    return dict(feature_to_in=(n_feature, n_inpt))


def exprs(inpt, feature_to_in, c_sparsity):
    """Return a dictionary containing various expressions for the model.

    Parameters
    ----------

    inpt : Theano variable
        Array of shape ``(n, d)`` where ``n`` is the number of samples and
        ``d`` is the dimensionality of the data.

    feature_to_in : Theano variable
        Array of shape ``(d, e)`` that is the transformation applied to the
        data.

    c_sparsity : float or Theano variable
        Coefficient for the sparsity penalty.

    Returns
    -------

    res : dict
        Dictionary containing expressions with various fields. The
        reconstruction is given as ``reconstruction`` and the corresponding loss
        coordinate wise, sample wise and for the whole data set via
        ``rec_loss_coord_wise``, ``rec_loss_sample_wise`` and ``rec_loss``.
        The sparsity loss is given accordingly as ``sparsity_loss_coord_wise``,
        ``sparsity_loss_sample_wise`` and ``sparsity_loss``. The complete loss
        is given via ``loss``.
    """
    feature_flat = T.vector('feature_flat')
    feature = feature_flat.reshape((inpt.shape[0], feature_to_in.shape[0]))

    reconstruction = T.dot(feature, feature_to_in)
    residual = inpt - reconstruction

    rec_loss_coord_wise = residual ** 2
    rec_loss_sample_wise = rec_loss_coord_wise.sum(axis=1)
    rec_loss = rec_loss_sample_wise.mean()

    sparsity_loss_coord_wise = T.sqrt((feature ** 2) + 1e-4)
    sparsity_loss_sample_wise = sparsity_loss_coord_wise.sum(axis=1)
    sparsity_loss = sparsity_loss_sample_wise.mean()

    loss = rec_loss + c_sparsity * sparsity_loss

    return get_named_variables(locals())
