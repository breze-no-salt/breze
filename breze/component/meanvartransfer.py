# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T


z = 1. / np.sqrt(2 * np.pi)


def normal_pdf(x, location=0, scale=1):
    Z = z / scale
    exp_arg = -((x - location) ** 2) / (2 * scale ** 2)
    return T.exp(exp_arg) * Z


def normal_cdf(x, location=0, scale=1):
    erf_arg = (x - location) / T.sqrt(2 * scale ** 2)
    return .5 * (1 + T.erf(erf_arg))
    pass


def rectifier(mean, var):
    std = T.sqrt(var)
    ratio = mean / std

    mean_ = normal_cdf(ratio) * mean + normal_pdf(ratio) * std

    A = mean * std * normal_pdf(ratio)
    B = (mean ** 2 + std ** 2) * normal_cdf(ratio)
    exp_of_squared = A + B

    var_ = exp_of_squared - mean_ ** 2
    return mean_, var_


def identity(mean, var):
    return mean, var
