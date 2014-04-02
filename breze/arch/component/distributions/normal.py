# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T


PI = np.array(np.pi, dtype=theano.config.floatX)
epsilon = np.array(1e-32, dtype=theano.config.floatX)


def pdf(sample, location=0, scale=1):
    """Return a theano expression representing the values of the probability
    density function of a Gaussian distribution.

    Parameters
    ----------

    sample : Theano variable
        Array of shape ``(n,)`` where ``n`` is the number of samples.

    location : Theano variable
        Scalar representing the mean of the distribution.

    scale : Theano variable
        Scalar representing the standard deviation of the distribution.


    Returns
    -------

    l : Theano variable
        Array of shape ``(n,)`` where each entry represents the density of the
        corresponding sample.


    Examples
    --------

    >>> import theano
    >>> import theano.tensor as T
    >>> import numpy as np
    >>> from breze.learn.utils import theano_floatx
    >>> sample, mean, std = T.vector(), T.scalar(), T.scalar()
    >>> p = pdf(sample, mean, std)
    >>> f_p = theano.function([sample, mean, std], p)

    >>> X, = theano_floatx(np.array([-1, 0, 1]))
    >>> ps = f_p(X, 0.1, 1.2)
    >>> np.allclose(ps,  [0.21840613,  0.33129956,  0.25094786])
    True
    """
    #location = T.cast(location, theano.config.floatX)
    SQRT_2_PI = np.sqrt(2 * PI)
    #SQRT_2_PI = T.cast(SQRT_2_PI, theano.config.floatX)

    divisor = 2 * scale ** 2 # + epsilon,
    #divisor = T.cast(divisor, theano.config.floatX)

    exp_arg = -((sample - location.dimshuffle('x')) ** 2) / divisor
    z = 1. / (SQRT_2_PI * scale + epsilon)

    return T.exp(exp_arg) * z


def cdf(sample, location=0, scale=1):
    """Return a theano expression representing the values of the cumulative
    density function of a Gaussian distribution.

    Parameters
    ----------

    sample : Theano variable
        Array of shape ``(n,)`` where ``n`` is the number of samples.

    location : Theano variable
        Scalar representing the mean of the distribution.

    scale : Theano variable
        Scalar representing the standard deviation of the distribution.

    Returns
    -------

    l : Theano variable
        Array of shape ``(n,)`` where each entry represents the cumulative
        density of the corresponding sample.


    Examples
    --------

    >>> import theano
    >>> import theano.tensor as T
    >>> import numpy as np
    >>> from breze.learn.utils import theano_floatx
    >>> sample, mean, std = T.vector(), T.scalar(), T.scalar()
    >>> c = cdf(sample, mean, std)
    >>> f_c = theano.function([sample, mean, std], c)

    >>> X, = theano_floatx(np.array([-1, 0, 1]))
    >>> cs = f_c(X, 0.1, 1.2)
    >>> np.allclose(cs, [0.17965868, 0.46679324, 0.77337265])
    True
    """
    location = T.cast(location, theano.config.floatX)
    scale = T.cast(scale, theano.config.floatX)

    div = T.sqrt(2 * scale ** 2 + epsilon)
    div = T.cast(div, theano.config.floatX)

    erf_arg = (sample - location) / div
    return .5 * (1 + T.erf(erf_arg + epsilon))
