# -*- coding: utf-8 -*-


from ...util import lookup, get_named_variables
from ...component import transfer as _transfer


def loss(code, transfer, prefix=''):
    """Return the ica loss of a code ``code`` given a density function.

    Parameters
    ----------

    code : Theano variable
        Array of shape ``(n, d)``, where ``n`` is then number of samples and
        ``d`` the dimensionality.

    transfer : string or function
        Density function. Either a function callable that returns a density for
        each entry of its argument or a string pointing at a function in
        ``breze.arch.component.transfer``.

    Returns
    -------

    res : dict
        Dictionary containing the loss coordinate wise, sample wise and
        completely. Corresponding keys are ``ica_loss_coord_wise``,
        ``ica_loss_sample_wise`` and ``ica_loss``.

    Examples
    --------

    >>> import theano.tensor as T
    >>> m = T.matrix()
    >>> loss(m, 'softabs', 'rica-')
    {'rica-ica_loss': Elemwise{true_div,no_inplace}.0, 'rica-code': <TensorType(float32, matrix)>, 'rica-ica_loss_sample_wise': Sum{1}.0, 'rica-ica_loss_coord_wise': Elemwise{sqrt,no_inplace}.0}
    """
    f_transfer = lookup(transfer, _transfer)
    ica_loss_coord_wise = f_transfer(code)
    ica_loss_sample_wise = ica_loss_coord_wise.sum(axis=1)
    ica_loss = ica_loss_sample_wise.mean()
    return get_named_variables(locals(), prefix=prefix)
