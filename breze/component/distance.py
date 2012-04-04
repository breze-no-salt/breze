# -*- coding: utf-8 -*-


import theano.tensor as T

import norm

from ..util import lookup


def cross_entropy(X, Y, axis=None):
    return -(X * T.log(Y)).sum(axis=axis)


def nominal_cross_entropy(X, Y, axis=None):
    return -(T.log(Y)[T.arange(X.shape[0]), X])


def bernoulli_cross_entropy(X, Y, axis=None):
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
    if isinstance(norm, (str, unicode)):
        norm_ = lookup(norm_, norm)
    Y = X if Y is None else Y
    diffs = X.dimshuffle(0, 1, 'x') - Y.dimshuffle('x', 1, 0)
    dist_comps = norm_(diffs, axis=1)
    return dist_comps


def nca(X, Y):
  """Return expression for the negative expected correctly classified
  points given a nearest neighbour classification method.

  As introduced in 'Neighbourhood Component Analysis'."""
  # Matrix of the distances of points.
  dist = distance_matrix(X)
  thisid = T.identity_like(dist)

  # Probability that a point is neighbour of another point based on
  # the distances.
  top = T.exp(-dist) + 1E-8 # Add a small constant for stability.
  bottom = (top - thisid * top).sum(axis=0)
  p = top / bottom

  # Create a matrix that matches same classes.
  sameclass = T.eq(distance_matrix(Y), 0) - thisid
  return -(p * sameclass).sum() / X.shape[0]
