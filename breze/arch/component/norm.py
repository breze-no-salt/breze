# -*- coding: utf-8 -*-

"""Module containing various norms."""


import theano.tensor as T


def l1(arr, axis=None):
    """Return the L1 norm of a tensor.

    Parameters
    ----------

    arr : Theano variable.
        The variable to calculate the norm of.
    axis : integer, optional [default: None]
        The sum will be performed along this axis. This makes it possible to
        calculate the norm of many tensors in parallel, given they are organized
        along some axis. If not given, the norm will be computed for the whole
        tensor.


    Returns
    -------

    res : Theano variable.
        If ``axis`` is ``None``, this will be a scalar. Otherwise it will be
        a tensor with one dimension less, where the missing dimension
        corresponds to ``axis``.


    Examples
    --------

    >>> v = T.vector()
    >>> this_norm = l1(v)

    >>> m = T.matrix()
    >>> this_norm = l1(m, axis=1)

    >>> m = T.matrix()
    >>> this_norm = l1(m)
    """
    return abs(arr).sum(axis=axis)


def soft_l1(inpt, eps=1e-8, axis=None):
    """Return a "soft" L1 norm of a tensor.

    The term "soft" is used because we are using :math:`\sqrt{x^2 + \epsilon}`
    in favor of :math:`|x|` which is not smooth at :math:`x=0`.

    Parameters
    ----------

    arr : Theano variable.
        The variable to calculate the norm of.

    eps : float, optional [default: 1e-8]
        Small offset to make the function more smooth.


    axis : integer, optional [default: None]
        The sum will be performed along this axis. This makes it possible to
        calculate the norm of many tensors in parallel, given they are organized
        along some axis. If not given, the norm will be computed for the whole
        tensor.

    Returns
    -------

    res : Theano variable.
        If ``axis`` is ``None``, this will be a scalar. Otherwise it will be
        a tensor with one dimension less, where the missing dimension
        corresponds to ``axis``.


    Examples
    --------

    >>> v = T.vector()
    >>> this_norm = soft_l1(v)

    >>> m = T.matrix()
    >>> this_norm = soft_l1(m, axis=1)

    >>> m = T.matrix()
    >>> this_norm = soft_l1(m)
    """

    return T.sqrt(inpt ** 2 + eps).sum(axis=axis)


def l2(arr, axis=None):
    """Return the L2 norm of a tensor.

    Parameters
    ----------

    arr : Theano variable.
        The variable to calculate the norm of.
    axis : integer, optional [default: None]
        The sum will be performed along this axis. This makes it possible to
        calculate the norm of many tensors in parallel, given they are organized
        along some axis. If not given, the norm will be computed for the whole
        tensor.

    Returns
    -------

    res : Theano variable.
        If ``axis`` is ``None``, this will be a scalar. Otherwise it will be
        a tensor with one dimension less, where the missing dimension
        corresponds to ``axis``.


    Examples
    --------

    >>> v = T.vector()
    >>> this_norm = l2(v)

    >>> m = T.matrix()
    >>> this_norm = l2(m, axis=1)

    >>> m = T.matrix()
    >>> this_norm = l2(m)
    """
    return T.sqrt((arr ** 2).sum(axis=axis) + 1e-8)


def lp(inpt, p, axis=None):
    """Return the Lp norm of a tensor.

    Parameters
    ----------

    arr : Theano variable.
        The variable to calculate the norm of.

    p : Theano variable or float.
        Order of the norm.

    axis : integer, optional [default: None]
        The sum will be performed along this axis. This makes it possible to
        calculate the norm of many tensors in parallel, given they are organized
        along some axis. If not given, the norm will be computed for the whole
        tensor.

    Returns
    -------

    res : Theano variable.
        If ``axis`` is ``None``, this will be a scalar. Otherwise it will be
        a tensor with one dimension less, where the missing dimension
        corresponds to ``axis``.


    Examples
    --------

    >>> v = T.vector()
    >>> this_norm = lp(v, .5)

    >>> m = T.matrix()
    >>> this_norm = lp(m, 3, axis=1)

    >>> m = T.matrix()
    >>> this_norm = lp(m, 4)
    """
    return ((inpt ** p).sum(axis=axis)) ** (1. / p)


def normalize(inpt, f_comp, axis, eps=1E-8):
    if axis not in (0, 1):
        raise ValueError('only axis 0 or 1 allowed')

    transformed = f_comp(inpt)
    this_norm = transformed.sum(axis=axis)
    if axis == 0:
        res = transformed / (this_norm + eps)
    elif axis == 1:
        res = (transformed.T / (this_norm + eps)).T

    return res


# TODO what about this? Is this a norm? Should it go?
def exp(inpt, axis=None):
    return T.exp(inpt).sum(axis=axis)
