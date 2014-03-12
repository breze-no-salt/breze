# -*- coding: utf-8 -*-

"""Module that contains functionality common to many other modules."""


import loss as loss_
from ..util import lookup, get_named_variables


def supervised_loss(target, prediction, loss, coord_axis=1, prefix=''):
    """Return a dictionary populated with several expressions for a supervised
    loss and corresponding targets and predictions.

    Parameters
    ----------

    target : Theano variable
        Array representing the target variables.

    prediction : Theano variable
        Array representing the predictions.

    loss : callable or string
        If a string, should index a member of :mod:`breze.arch.component.loss`.
        If a callable, has to be a of the form described in
        :mod:`breze.arch.component.loss`.

    coord_axis : integer, optional [default: 1]
        Axis aong which the coordinates of single sample are stored. I.e. not
        the sample axis or some spatial axis.

    prefix : string, optional [default: '']
        Each key in the resulting dictionary will be prefixed with ``prefix``.

    Returns
    -------

    res : dict
        Dictionary containing the expressions. See example for keys.

    Examples
    --------

    >>> import theano.tensor as T
    >>> prediction, target = T.matrix('prediction'), T.matrix('target')
    >>> from breze.arch.component.loss import squared
    >>> loss_dict = supervised_loss(target, prediction, squared,
    ...   prefix='mymodel-')
    >>> sorted(loss_dict.items())
    [('mymodel-loss', Elemwise{true_div,no_inplace}.0), ('mymodel-loss_coord_wise', Elemwise{pow,no_inplace}.0), ('mymodel-loss_sample_wise', Sum{1}.0), ('mymodel-prediction', prediction), ('mymodel-target', target)]
    """
    f_loss = lookup(loss, loss_)
    loss_coord_wise = f_loss(target, prediction)
    loss_sample_wise = loss_coord_wise.sum(axis=coord_axis)
    loss = loss_sample_wise.mean()

    return get_named_variables(locals(), prefix=prefix)


def unsupervised_loss(output, loss, coord_axis=1, prefix=''):
    """Return a dictionary populated with several expressions for a
    unsupervised loss and corresponding output.

    Parameters
    ----------

    output : Theano variable
        Array representing the predictions.

    loss : callable or string
        If a string, should index a member of :mod:`breze.arch.component.loss`.
        If a callable, has to be a of the form described in
        :mod:`breze.arch.component.loss`.

    coord_axis : integer, optional [default: 1]
        Axis aong which the coordinates of single sample are stored. I.e. not
        the sample axis or some spatial axis.

    prefix : string, optional [default: '']
        Each key in the resulting dictionary will be prefixed with ``prefix``.

    Returns
    -------

    res : dict
        Dictionary containing the expressions. See example for keys.

    Examples
    --------

    >>> import theano.tensor as T
    >>> output = T.matrix('output')
    >>> my_loss = lambda x: abs(x)
    >>> loss_dict = unsupervised_loss(output, my_loss, prefix='$')
    >>> sorted(loss_dict.items())
    [('$loss', Elemwise{true_div,no_inplace}.0), ('$loss_coord_wise', Elemwise{abs_,no_inplace}.0), ('$loss_sample_wise', Sum{1}.0), ('$output', output)]
    """
    f_loss = lookup(loss, loss_)
    loss_coord_wise = f_loss(output)
    loss_sample_wise = loss_coord_wise.sum(axis=coord_axis)
    loss = loss_sample_wise.mean()

    return get_named_variables(locals(), prefix=prefix)
