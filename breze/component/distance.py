# -*- coding: utf-8 -*-


import theano.tensor as T

import norm

from ..util import lookup


def neg_cross_entropy(target, prediction, axis=None, eps=1e-24):
    return -(target * T.log(prediction + eps)).sum(axis=axis)


def nominal_neg_cross_entropy(target, prediction, axis=None):
    return -(T.log(prediction)[T.arange(target.shape[0]), target])


def bernoulli_neg_cross_entropy(X, Y, axis=None):
    return -(X * T.log(Y) + (1 - X) * T.log(1 - Y)).sum(axis=axis)


def squared(X, Y, axis=None):
    diff = X - Y
    return norm.l2(diff, axis=axis)


def absolute(X, Y, axis=None):
    diff = X - Y
    return norm.l1(diff, axis=axis)


def bernoulli_kl(X, Y):
    """
    Kullback-Leibler divergence between two
    bernoulli random variables _X_ and _Y_.
    """
    return (X * T.log(X / Y) + (1 - X) * T.log((1 - X) / (1 - Y))).sum()


def discrete_entropy(X, axis=None):
    return -(X * T.log(X)).sum(axis=axis)


def distance_matrix(X, Y=None, norm_=norm.l2):
    """Return an expression containing the distances given the norm."""
    if isinstance(norm_, (str, unicode)):
        norm_ = lookup(norm_, norm)
    Y = X if Y is None else Y
    diffs = X.dimshuffle(0, 1, 'x') - Y.dimshuffle('x', 1, 0)
    dist_comps = norm_(diffs, axis=1)
    return dist_comps


def nca(target, embedding):
     """Return expression for the negative expected correctly classified
     points given a nearest neighbour classification method.

     As introduced in 'Neighbourhood Component Analysis'."""
     # Matrix of the distances of points.
     dist = distance_matrix(embedding)
     thisid = T.identity_like(dist)

     # Probability that a point is neighbour of another point based on
     # the distances.
     top = T.exp(-dist) + 1E-8 # Add a small constant for stability.
     bottom = (top - thisid * top).sum(axis=0)
     p = top / bottom

     # Create a matrix that matches same classes.
     sameclass = T.eq(distance_matrix(target), 0) - thisid
     return -(p * sameclass).sum() / embedding.shape[0]
