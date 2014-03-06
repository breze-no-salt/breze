# -*- coding: utf-8 -*-

"""Module that keeps various transfer functions as used in the context of
neural networks."""


import theano.tensor as T
import theano.tensor.nnet


def tanh(inpt):
    """Tanh activation function.

    Parameters
    ----------

    inpt : Theano variable
        Input to be transformed.

    Returns
    -------

    output : Theano variable
        Transformed output. Same shape as ``inpt``.
    """
    return T.tanh(inpt)


def tanhplus(inpt):
    """Tanh with added linear activation function.

    .. math::

       f(x) = tanh(x) + x


    Parameters
    ----------

    inpt : Theano variable
        Input to be transformed.

    Returns
    -------

    output : Theano variable
        Transformed output. Same shape as ``inpt``.
    """
    return T.tanh(inpt) + inpt


def sigmoid(inpt):
    """Sigmoid activation function.

    .. math::

       f(x) = {1 \over 1 + \exp(-x)}


    Parameters
    ----------

    inpt : Theano variable
        Input to be transformed.

    Returns
    -------

    output : Theano variable
        Transformed output. Same shape as ``inpt``.
    """
    return T.nnet.sigmoid(inpt)


def rectifier(inpt):
    """Rectifier activation function.

    .. math::

       f(x) = \max(0, x)


    Parameters
    ----------

    inpt : Theano variable
        Input to be transformed.

    Returns
    -------

    output : Theano variable
        Transformed output. Same shape as ``inpt``.
    """
    return T.maximum(inpt, 0)


def softplus(inpt):
    """Soft plus activation function.

    Smooth approximation to ``rectifier``.

    .. math::

       f(x) = \log (1 + \exp(x))


    Parameters
    ----------

    inpt : Theano variable
        Input to be transformed.

    Returns
    -------

    output : Theano variable
        Transformed output. Same shape as ``inpt``.
    """
    return T.log(1 + T.exp(inpt))


def identity(inpt):
    """Identity activation function.

    .. math::

       f(x) = x


    Parameters
    ----------

    inpt : Theano variable
        Input to be transformed.

    Returns
    -------

    output : Theano variable
        Transformed output. Same shape as ``inpt``.
    """


def softmax(inpt):
    """Softmax activation function.

    .. math::

       f(x_i) = {\exp(x_i) \over \sum_j \exp(x_j)}

    Here, the index runs over the columns of ``inpt``.

    Numerical stable version that subtracts the maximum of each row from all of
    its entries.

    Wrapper for ``theano.nnet.softmax``.


    Parameters
    ----------

    inpt : Theano variable
        Array of shape ``(n, d)``. Input to be transformed.

    Returns
    -------

    output : Theano variable
        Transformed output. Same shape as ``inpt``.
    """
    return T.nnet.softmax(inpt)


def softsign(inpt):
    """Softsign activation function.

    .. math::

       f(x) = {x \over 1 + |x|}

    Parameters
    ----------

    inpt : Theano variable
        Input to be transformed.

    Returns
    -------

    output : Theano variable
        Transformed output. Same shape as ``inpt``.
    """
    return inpt / (1 + abs(inpt))


# TODO move this into a different module, e.g. "densities" or so.

def logproduct_of_t(inpt):
    return T.log(1 + inpt ** 2)


def logcosh(inpt):
    return T.log(T.cosh(inpt))


def softabs(inpt, eps=1E-5):
    return T.sqrt(inpt ** 2 + eps)
