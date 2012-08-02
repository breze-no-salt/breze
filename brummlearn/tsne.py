# -*- coding: utf-8 -*-

"""Stochastic neighbourhood embedding with Student's t."""

import itertools

import climin
import numpy as np
import theano
import theano.tensor as T

from breze.component.distance import distance_matrix
from climin.base import Minimizer
from scipy.spatial.distance import pdist, squareform, cdist


def zero_diagonal(X):
    """Given a square matrix `X`, return a theano variable with the diagonal of
    `X` set to zero."""
    thisid = T.identity_like(X)
    return (X - thisid * X)


def neighbour_probabilities(X, target_pplx):
    """Return a square matrix containing probabilities that points given by `X`
    in the data are neighbours."""
    N = X.shape[0]
    
    # Calculate the distances.
    dists = squareform(pdist(X, 'euclidean')**2)

    # Parametrize in the log domain for positive standard deviations.
    logstds = np.ones(X.shape[0]) * 2

    # Do a binary search for good logstds leading to the desired perplexities.

    minimums = np.empty(X.shape[0])
    minimums[...] = -np.inf
    maximums = np.empty(X.shape[0])
    maximums[...] = np.inf

    for j in range(50):
        # Calculate perplexities.
        vars = np.exp(logstds)**2
        inpt_top = np.exp(-dists / 2 / vars)
        inpt_top[range(N), range(N)] = 0
        inpt_bottom = inpt_top.sum(axis=0)

        # If we don't add a small term, the logarithm will make NaNs.
        p_inpt_nb_cond = inpt_top / inpt_bottom + 1E-12
        entropy = -(p_inpt_nb_cond * np.log(p_inpt_nb_cond)).sum(axis=0)
        pplxs = np.exp(entropy)

        diff = pplxs - target_pplx
        for j in range(N):
            if abs(diff[j]) < 1e-5:
                continue
            elif diff[j] < 0:
                minimums[j] = logstds[j]
                if maximums[j] == np.inf:
                    logstds[j] *= 2
                else:
                    logstds[j] = (logstds[j] + maximums[j]) / 2
            else:
                maximums[j] = logstds[j]
                if minimums[j] == -np.inf:
                    logstds[j] /= 2
                else:
                    logstds[j] = (logstds[j] + minimums[j]) / 2

    # Calculcate p matrix once more and return it.
    vars = np.exp(logstds)**2
    inpt_top = np.exp(-dists / 2 / vars)
    inpt_top[range(N), range(N)] = 0
    inpt_bottom = inpt_top.sum(axis=0)
    p_inpt_nb_cond = inpt_top / inpt_bottom + 1E-12

    # Symmetrize.
    p_ji = (p_inpt_nb_cond + p_inpt_nb_cond.T)

    # We don't normalize correctly here. But that does not matter, since we 
    # normalize q wrongly in the same way.
    p_ji /= p_ji.sum()

    return p_ji


def build_loss(embeddings):
    """Return a pair (loss, p) given a theano shared variable representing the
    `embeddings`.
    
    `loss` is a theano variable for the loss. `p` is a symbolic variable
    representing the target neighbour probabilities on which the loss depends.
    """
    # Probability that two points are neighbours in the embedding space.
    emb_dists = distance_matrix(embeddings)
    emb_top = zero_diagonal(1 / (1 + emb_dists)) + 1E-12
    emb_bottom = emb_top.sum(axis=0)
    q = emb_top / emb_bottom

    # Incorrect normalization which does not matter since we normalize p i 
    # the same way.
    q /= q.sum()
    q = T.clip(q, 1E-12, 1)

    p_ji_var = T.matrix('neighbour_probabilities')

    # t-distributed stochastic neighbourhood embedding loss.
    loss = (p_ji_var * T.log(p_ji_var / q)).sum()

    return loss, p_ji_var



# Custom Minimizer for TSNE using a learning rate schedule.
class TsneMinimizer(Minimizer):

    def __init__(self, wrt, fprime, steprate, momentum, min_gain=1E-2,
                 args=None, logfunc=None):
        super(TsneMinimizer, self).__init__(wrt, args=args, logfunc=logfunc)

        self.fprime = fprime
        self.steprate = steprate
        self.momentum = momentum
        self.min_gain = min_gain

    def __iter__(self):
        step_m1 = np.zeros(self.wrt.shape[0])
        gain = np.ones(self.wrt.shape[0])

        for i, (args, kwargs) in enumerate(self.args):
            gradient = self.fprime(self.wrt, *args, **kwargs)
            gain = (gain + 0.2) * ((gradient > 0) != (step_m1 > 0)) # different signs
            gain += 0.8 * ((gradient > 0) == (step_m1 > 0))         # same signs
            gain[gain < self.min_gain] = self.min_gain
            step = self.momentum * step_m1 
            step -= self.steprate * gradient * gain
            self.wrt += step
            step_m1 = step
            yield dict(gradient=gradient, gain=gain, args=args, kwargs=kwargs,
                       n_iter=i, step=step)


def tsne(X, low_dim, perplexity=40, early_exaggeration=50, max_iter=1000):
    if early_exaggeration < 0:
        raise ValueError("early_exaggeration has to be non negative")
    if max_iter < 0:
        raise ValueError("max_iter has to be non negative")

    # Define embeddings shared variable and initialize randomly.
    embeddings_flat = theano.shared(np.random.normal(0, 1e-4, X.shape[0] * low_dim))
    embeddings = embeddings_flat.reshape((X.shape[0], low_dim))
    embeddings_data = embeddings_flat.get_value(
        borrow=True, return_internal_type=True)
    embeddings_sub = T.vector()

    # Calculate the target neighbour probabilities.
    p_ji = neighbour_probabilities(X, perplexity)

    # Create loss expression and its gradient.
    loss, p_ji_var = build_loss(embeddings)
    d_loss_wrt_embeddings_flat = T.grad(loss, embeddings_flat)

    # Compile functions.
    givens = {embeddings_flat: embeddings_sub}
    f_loss = theano.function(
        [embeddings_sub, p_ji_var], loss, givens=givens)
    f_d_loss = theano.function(
        [embeddings_sub, p_ji_var], d_loss_wrt_embeddings_flat, givens=givens)

    # Optimize with early exaggeration.
    p_ji_exaggerated = p_ji * 4
    p_ji_exaggerated = np.clip(p_ji_exaggerated, 1E-12, 1)
    args = (([p_ji_exaggerated], {}) for _ in itertools.count())

    opt = TsneMinimizer(embeddings_data, f_d_loss, args=args, momentum=0.5,
                        steprate=500, min_gain=0.01)

    prev = embeddings_data.copy()
    for i, info in enumerate(opt):
        if i > 20:
            opt.momentum = 0.8
        print i, f_loss(embeddings_data, p_ji * 4), embeddings_data.sum()
        update = embeddings_data - prev
        print update.mean(), update.max(), update.min()
        if i + 1 == early_exaggeration:
            break

    # Optimize with no lying about p values.
    args = (([p_ji], {}) for _ in itertools.count())
    opt = TsneMinimizer(embeddings_data, f_d_loss, args=args, momentum=0.8,
                        steprate=500, min_gain=0.01)
    for i, info in enumerate(opt):
        print i, f_loss(embeddings_data, p_ji), embeddings_data.sum()
        if i + 1 == max_iter - early_exaggeration:
            break

    return embeddings_data.reshape(X.shape[0], low_dim)
