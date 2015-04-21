# -*- coding: utf-8 -*-


import theano.tensor as T

from ...util import lookup, get_named_variables
from ...component import transfer as _transfer, norm


# TODO docstring examples


def parameters(n_inpt, n_output):
    """Return the parameter specification dictionary for a sparse filtering
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
    """
    return dict(in_to_out=(n_inpt, n_output))


# TODO add prefix

def loss(output, density):
    """Return the sparse filtering loss of a code ``inpt`` given a density
    function.

    Parameters
    ----------

    inpt : Theano variable
        Array of shape ``(n, d)``, where ``n`` is then number of samples and
        ``d`` the dimensionality.

    density : string or function
        Density function. Either a function callable that returns a density for
        each entry of its argument or a string pointing at a function in
        ``breze.arch.component.transfer``.

    Returns
    -------

    res : dict
        Dictionary containing the loss sample wise and
        completely, ``loss_sample_wise`` and ``loss`` respectively. Also,
        column normalized features ``col_normalized``, row normalized features
        ``row_normalized``, and the output after applying the density function
        ``output_post``.
    """
    f_density = lookup(density, _transfer)
    output_post = f_density(output)

    col_normalized = T.sqrt(
        norm.normalize(output_post, lambda x: x ** 2, axis=0) + 1E-8)
    row_normalized = T.sqrt(
        norm.normalize(col_normalized, lambda x: x ** 2, axis=1) + 1E-8)

    loss_sample_wise = row_normalized.sum(axis=1)
    loss = loss_sample_wise.mean()

    return get_named_variables(locals())
