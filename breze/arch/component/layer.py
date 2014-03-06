# -*- coding: utf-8 -*-

import theano.tensor as T

from ..util import lookup, get_named_variables

import transfer
import corrupt


def simple(inpt, weights, bias, out_transfer, p_dropout=0, prefix=''):
    """Return a dictionary containing computations from a simple layer.

    The layer has the following form

    .. math::
        f((x \cdot d) ^TW + b),

    where :math:`f` corresponds to ``transfer``, :math:`x` to ``input``,
    :math:`\cdot` indicates the element-wise product, :math:`d` is a vector
    of Bernoulli samples with parameter ``p_dropout``, :math:`W` is the weight
    matrix ``weights`` and :math:`b` is the ``bias``.

    Parameters
    ----------

    inpt : Theano variable
        Array of shape ``(n, d)``.

    weights : Theano variable
        Array of shape ``(d, e)``.

    bias : Theano variable
        Array of shape ``(e,)``.

    transfer : function or string
        If a function should given a Theano variable return a Theano variable of
        the same shape. If string, is used to get a transfer function from
        ``breze.arch.component.transfer``.

    p_dropout : Theano scalar or float
        Needs to be in (0, 1). Indicates the probability that an input is set to
        zero.

    prefix : string, optional [default: '']
        Each enty in the returned dictionary will be prefixed with this.

    Returns
    -------

    d : dict
        Has the following entries: ``output_in``, activation before application
        of ``transfer``. ``output``, activation after application of
        ``transfer``.
    """

    output_in = T.dot(inpt, weights) + bias
    f_output = lookup(out_transfer, transfer)
    output = f_output(output_in)
    if p_dropout != 0:
        output = corrupt.mask(output, p_dropout)

    return get_named_variables(locals(), prefix=prefix)
