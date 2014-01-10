# -*- coding: utf-8 -*-

import theano.tensor as T

from ..component import misc
from ..util import lookup, get_named_variables
from linear import parameters, exprs as linear_exprs


def loss(posterior, pars_to_penalize, c_rim):
    marginal = posterior.mean(axis=0)
    cond_entropy = misc.discrete_entropy(posterior, axis=1).mean()
    entropy = misc.discrete_entropy(marginal)

    nmi = cond_entropy - entropy

    n_samples = posterior.shape[0]
    penalties = [(i ** 2).sum() / n_samples for i in pars_to_penalize]
    penalty = sum(penalties)

    loss = nmi + c_rim * penalty

    return get_named_variables(locals())
