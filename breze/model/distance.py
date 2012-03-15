# -*- coding: utf-8 -*-


import theano.tensor as T

from norm import l1, l2


def bernoulli_cross_entropy(X, Y, axis=None):
    return -(X * T.log(Y)).sum(axis=axis)


def euclidean(X, Y, axis=None):
    diff = X - Y
    return l2(diff, axis=axis)


def manhattan(X, Y, axis=None):
    diff = X - Y
    return l1(diff, axis=axis)
