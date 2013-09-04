"""This module contains functionality to corrupt Theano variables
with noise."""

import theano
import theano.tensor as T


def gaussian_perturb(x, std, rng=None):
    """Return a Theano variable which is ``x + e`` where ``e`` is Gaussian noise
    with a standard deviation of `std`."""
    if rng is None:
        rng = T.shared_randomstreams.RandomStreams()
    noise = rng.normal(size=x.shape, std=std)
    noise = T.cast(noise, theano.config.floatX)
    return x + noise


def mask(x, p, rng=None):
    """Return a Theano variable which is ``x`` with elements of it set to zero
    with a probability of ``p``."""
    if rng is None:
        rng = T.shared_randomstreams.RandomStreams()
    this_mask = T.cast(rng.binomial(size=x.shape, p=p), theano.config.floatX)
    return x * this_mask
