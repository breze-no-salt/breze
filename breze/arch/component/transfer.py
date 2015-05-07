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
    return inpt


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


def diag_gauss(inpt):
    """Transfer function to turn an arary into sufficient statistics of a
    diagonal Gaussian.

    The first half of the input will be left unchanged, the second will be
    squared. the "split" into halves is performed along the second axis.

    Parameters
    ----------

    inpt : Theano tensor
        Array of shape ``(n, d)`` or ``(t, n, d)``.

    Returns
    -------

    output : Theano variable.
        Transformed input. Same shape as ``inpt``.
    """
    half = inpt.shape[-1] // 2
    if inpt.ndim == 3:
        mean, var = inpt[:, :, :half], inpt[:, :, half:]
        res = T.concatenate([mean, var ** 2 + 1e-8], axis=2)
    else:
        mean, var = inpt[:, :half], inpt[:, half:]
        res = T.concatenate([mean, var ** 2 + 1e-8], axis=1)
    return res


def diag_laplace(inpt):
    """Transfer function to turn an arary into sufficient statistics of a
    diagonal Laplace distribution.

    The "split" into halves is performed along the second axis.

    Parameters
    ----------

    inpt : Theano tensor
        Array of shape ``(n, d)`` or ``(t, n, d)``.

    Returns
    -------

    output : Theano variable.
        Transformed input. Same shape as ``inpt``.
    """
    half = inpt.shape[-1] // 2
    if inpt.ndim == 3:
        mean, b = inpt[:, :, :half], inpt[:, :, half:]
        res = T.concatenate([mean, abs(b)], axis=2)
    else:
        mean, b = inpt[:, :half], inpt[:, half:]
        res = T.concatenate([mean, abs(b)], axis=1)
    print 'diag_laplace: mean' +str(mean.shape)
    print 'diag_laplace: b' +str(b.shape)
    return res


# TODO move this into a different module, e.g. "densities" or so.

def logproduct_of_t(inpt):
    return T.log(1 + inpt ** 2)


def logcosh(inpt):
    return T.log(T.cosh(inpt))


def softabs(inpt, eps=1E-5):
    return T.sqrt(inpt ** 2 + eps)
