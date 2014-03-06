# -*- coding: utf-8 -*-

"""Module that offers functionality for RIM.

The relevant reference is

    Gomes, Ryan, Andreas Krause, and Pietro Perona.
    "Discriminative Clustering by Regularized Information Maximization."
    NIPS. 2010.
"""


import theano.tensor as T

from ..component import misc
from ..util import lookup, get_named_variables
from linear import parameters, exprs as linear_exprs


def loss(posterior, pars_to_penalize, c_rim):
    """Return the Regularized Information Maximization (RIM) loss of a set of
    categorical distributions.

    Parameters
    ----------

    posterior : Theano variable
        Array of the shape ``(n, d)``. The array describes ``n`` catgorical
        distributions over ``d`` categories. Each row has thus to sum up to 1
        and each entry has to be non-negative.

    pars_to_penalize : list of Theano variables
        Each of the items is a Theano variable that is being penalized with its
        squared L2 norm.

    c_rim : float
        Weight of the L2 penalties.
    """
    marginal = posterior.mean(axis=0)
    cond_entropy = misc.discrete_entropy(posterior, axis=1).mean()
    entropy = misc.discrete_entropy(marginal)

    nmi = cond_entropy - entropy

    n_samples = posterior.shape[0]
    penalties = [(i ** 2).sum() / n_samples for i in pars_to_penalize]
    penalty = sum(penalties)

    loss = nmi + c_rim * penalty

    return get_named_variables(locals())
