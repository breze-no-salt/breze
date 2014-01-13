# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def multinomial_weights(inpt):
    # Numerical stability.
    inpt = T.maximum(inpt, -10)
    inpt_normed = inpt - inpt.min(axis=0).dimshuffle('x', 0)
    inpt_normed = T.minimum(inpt_normed, 100)
    return T.exp(inpt_normed) / (T.exp(inpt_normed).sum(axis=0) + 1e-4)


def weighted_pooling(inpt):
    # First do a stable softmax over time.
    inpt_flat = inpt.reshape((inpt.shape[0], inpt.shape[1] * inpt.shape[2]))
    p = multinomial_weights(inpt_flat)

    inpt_flat *= p
    res_flat = inpt_flat.sum(axis=0)
    return res_flat.reshape((inpt.shape[1], inpt.shape[2]))


def pooling_layer(inpt, typ):
    if typ == 'mean':
        output = T.mean(inpt, axis=0)
    elif typ == 'sum':
        output = T.sum(inpt, axis=0)
    elif typ == 'prod':
        output = T.prod(inpt, axis=0)
    elif typ == 'min':
        output = T.min(inpt, axis=0)
    elif typ == 'max':
        output = T.max(inpt, axis=0)
    elif typ == 'last':
        output = inpt[-1]
    elif typ == 'stochastic':
        output = stochastic_pooling(inpt)
    else:
        raise ValueError('unknown pooling operator %s' % typ)
    return output


def stochastic_pooling(inpt, rng=None):
    if rng is None:
        srng = RandomStreams()

    # First do a stable softmax over time.
    inpt_flat = inpt.reshape((inpt.shape[0], inpt.shape[1] * inpt.shape[2]))
    p = multinomial_weights(inpt_flat)

    # Sum up the probabilities giving the cdf.
    cumulative, _ = theano.scan(
        lambda prior_result, c: prior_result + c,
        p,
        outputs_info=T.zeros_like(p[0]))

    # Draw Uniformly and check into which interval of the cdf the sample falls.
    u = srng.uniform(size=inpt_flat.shape)[0, :]
    picks = T.eq((u < cumulative), 1)
    idxs = T.argmax(picks, axis=0)

    # Return that sample.
    res_flat = inpt_flat[idxs, T.arange(0, idxs.shape[0])]
    return res_flat.reshape((inpt.shape[1], inpt.shape[2]))
