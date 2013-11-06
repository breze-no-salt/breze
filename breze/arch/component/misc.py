# -*- coding: utf-8 -*-


import theano.tensor as T

import norm as norm_
from ..util import lookup


def pairwise_diff(X, Y=None):
    Y = X if Y is None else Y
    diffs = X.T.dimshuffle(1, 0, 'x') - Y.T.dimshuffle('x', 0, 1)
    return diffs


def distance_matrix(X, Y=None, norm=norm_.l2):
    """Return an expression containing the distances given the norm."""
    diff = pairwise_diff(X, Y)
    return distance_matrix_by_diff(diff, norm=norm)


def distance_matrix_by_diff(diff, norm=norm_.l2):
    if isinstance(norm, (str, unicode)):
        norm = lookup(norm, norm_)
    dist_comps = norm(diff, axis=1)
    return dist_comps


def discrete_entropy(X, axis=None):
    X = T.minimum(1, X + 1e-8)
    return -(X * T.log(X)).sum(axis=axis)


def project_into_l2_ball(x, radius):
    # Distinguish whether we are given a single or many vectors to work upon.
    batch = x.ndim == 2
    if not batch:
        x = T.shape_padleft(x)

    lengths = T.sqrt((x ** 2).sum(axis=1)).dimshuffle(0, 'x')
    x = T.switch(lengths > radius, x / lengths * radius, x)

    if not batch:
        x = x[0]

    return x
