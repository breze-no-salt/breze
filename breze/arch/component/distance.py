# -*- coding: utf-8 -*-


import theano.tensor as T

import norm

from ..util import lookup


# TODO which modules are using this? Is there a better place to put these?
# Also, no documentation.


def bern_nce(X, Y):
    return -(X * T.log(Y) + (1 - X) * T.log(1 - Y)).sum(axis=axis)


def bernoulli_kl(X, Y, axis=None):
    """
    Kullback-Leibler divergence between two
    bernoulli random variables _X_ and _Y_.
    """
    return (X * T.log(X / Y) + (1 - X) * T.log((1 - X) / (1 - Y))
        ).sum(axis=axis)
