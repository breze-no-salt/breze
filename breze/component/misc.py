# -*- coding: utf-8 -*-


import theano.tensor as T

import norm
from ..util import lookup


def distance_matrix(X, Y=None, norm_=norm.l2):
    """Return an expression containing the distances given the norm."""
    if isinstance(norm_, (str, unicode)):
        norm_ = lookup(norm_, norm)
    Y = X if Y is None else Y
    diffs = X.T.dimshuffle(1, 0, 'x') - Y.T.dimshuffle('x', 0, 1)
    dist_comps = norm_(diffs, axis=1)
    return dist_comps


def discrete_entropy(X, axis=None):
    X = T.minimum(1, X + 1e-8)
    return -(X * T.log(X)).sum(axis=axis)
