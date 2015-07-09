
# -*- coding: utf-8 -*-


import loss as loss_
from ...util import lookup, get_named_variables


def supervised_loss(target, prediction, loss, coord_axis=1,
                    imp_weight=False,
                    prefix=''):
    """Return a dictionary populated with several expressions for a supervised
    loss and corresponding targets and predictions.

    Version for variance propagation, where the prediction is not only a point
    but a mean with a variance.

    Parameters
    ----------

    target : Theano variable
        Array representing the target variables. Has size ``d`` along the
        coordinate axis ``coord_axis``.

    prediction : Theano variable
        Array representing the predictions. Has size ``2 * d`` along the
        coordinate axis, where the first half corresponds to the mean and the
        second half to the variance of the prediction.

    loss : callable or string
        If a string, should index a member of :mod:`breze.arch.component.loss`.
        If a callable, has to be a of the form described in
        :mod:`breze.arch.component.varprop.loss`.

    coord_axis : integer, optional [default: 1]
        Axis aong which the coordinates of single sample are stored. I.e. not
        the sample axis or some spatial axis.

    imp_weight : Theano variable, float or boolean, optional [default: False]
        Importance weights for the loss. Will be multiplied to the coordinate
        wise loss.

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
    >>> from breze.arch.component.varprop.loss import diag_gaussian_nll
    >>> loss_dict = supervised_loss(target, prediction, diag_gaussian_nll,
    ...   prefix='mymodel-')
    >>> sorted(loss_dict.items())  # doctest: +ELLIPSIS
    [('mymodel-loss', ...), ('mymodel-loss_coord_wise', ...), ('mymodel-loss_sample_wise', ...), ('mymodel-prediction', prediction), ('mymodel-target', target)]
    """
    f_loss = lookup(loss, loss_)
    loss_coord_wise = f_loss(target, prediction)
    loss_coord_wise *= imp_weight if imp_weight else 1
    try:
        loss_sample_wise = loss_coord_wise.sum(axis=coord_axis)
    except ValueError:
        #we do not have enough dimensions, the loss is not coordinate-wise
        loss_sample_wise = loss_coord_wise
    if imp_weight:
        loss = loss_coord_wise.sum(axis=None) / imp_weight.sum(axis=None)
    else:
        loss = loss_sample_wise.mean()
    return get_named_variables(locals(), prefix=prefix)


def unsupervised_loss(output, loss, coord_axis=1, prefix=''):
    """Return a dictionary populated with several expressions for a
    unsupervised loss and corresponding output.

    Version for variance propagation, where the prediction is not only a point
    but a mean with a variance.

    Parameters
    ----------

    output : Theano variable
        Array representing the output of the model. Has size ``2 * d`` along
        the coordinate axis, where the first half corresponds to the mean and
        the second half to the variance of the prediction.

    loss : callable or string
        If a string, should index a member of :mod:`breze.arch.component.loss`.
        If a callable, has to be a of the form described in
        :mod:`breze.arch.component.varprop.loss`.

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
    >>> sorted(loss_dict.items()) # doctest: +ELLIPSIS
    [('$loss', ...), ('$loss_coord_wise', ...), ('$loss_sample_wise', ...), ('$output', ...)]
    """
    f_loss = lookup(loss, loss_)
    loss_coord_wise = f_loss(output)
    loss_sample_wise = loss_coord_wise.sum(axis=coord_axis)
    loss = loss_sample_wise.mean()

    return get_named_variables(locals(), prefix=prefix)
