import numpy as np
import theano
import theano.tensor as T


PI = np.array(np.pi, dtype=theano.config.floatX)


def pdf(x, location=0, scale=1):
    location = T.cast(location, theano.config.floatX)
    SQRT_2_PI = np.sqrt(2 * PI)
    SQRT_2_PI = T.cast(SQRT_2_PI, theano.config.floatX)

    divisor = 2 * scale ** 2 + epsilon,
    divisor = T.cast(divisor, theano.config.floatX)

    exp_arg = -((x - location) ** 2) / divisor
    z = 1. / (SQRT_2_PI * scale + epsilon)

    return T.exp(exp_arg) * z


def cdf(x, location=0, scale=1):
    location = T.cast(location, theano.config.floatX)
    scale = T.cast(scale, theano.config.floatX)

    div = T.sqrt(2 * scale ** 2 + epsilon)
    div = T.cast(div, theano.config.floatX)

    erf_arg = (x - location) / div
    return .5 * (1 + T.erf(erf_arg + epsilon))
