# -*- coding: utf-8 -*-


import theano.tensor as T

# TODO many of these are not norms. They should be renamed or so.
# TODO also, document


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
    """

    return abs(arr).sum(axis=axis)


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
    """
    return ((inpt ** p).sum(axis=axis)) ** (1. / p)


def exp(inpt, axis=None):
    return T.exp(inpt).sum(axis=axis)


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


def soft_l1(inpt, axis=None):
    return T.sqrt(inpt**2 + 1e-8).sum(axis=axis)
