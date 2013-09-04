# -*- coding: utf-8 -*-

"""tSNE is a method to find low dimensional representations of data.
The "crowding" problem is essentially solved, because the error function
of tSNE favors solutions which represent the data locally.

For details, see Laurens van der Maaten's page on tSNE at
http://homepage.tudelft.nl/19j49/t-SNE.html.
"""

import itertools

import numpy as np
import theano
import theano.tensor as T

from breze.arch.component.misc import distance_matrix
from climin.base import Minimizer


def zero_diagonal(X):
    """Given a square matrix `X`, return a theano variable with the diagonal of
    `X` set to zero."""
    thisid = T.identity_like(X)
    return (X - thisid * X)


def euc_dist(X, Y, squared=True):
    """
    Compute distances between
    rows in X and rows in Y.

    See http://blog.smola.org/post/969195661/in-praise-of-the-second-binomial-formula
    """
    if X is Y:
        Xsq = (X ** 2).sum(axis=1)
        Ysq = Xsq[np.newaxis, :]
        Xsq = Xsq[:, np.newaxis]
    else:
        Xsq = (X ** 2).sum(axis=1)[:, np.newaxis]
        Ysq = (Y ** 2).sum(axis=1)[np.newaxis, :]
    distances = Xsq + Ysq - 2 * np.dot(X, Y.T)
    if squared:
        return distances
    else:
        return np.sqrt(distances)


def neighbour_probabilities(X, target_pplx):
    """Return a square matrix containing probabilities that points given by `X`
    in the data are neighbours."""
    N = X.shape[0]

    # Calculate the distances.
    dists = euc_dist(X, X)

    # Parametrize in the log domain for positive standard deviations.
    precisions = np.ones(X.shape[0])

    # Do a binary search for good logstds leading to the desired perplexities.
    minimums = np.empty(X.shape[0])
    minimums[...] = -np.inf
    maximums = np.empty(X.shape[0])
    maximums[...] = np.inf

    target_entropy = np.log(target_pplx)
    for i in range(50):
        # Calculate perplexities.
        inpt_top = np.exp(-dists * precisions)
        inpt_top[range(N), range(N)] = 0
        inpt_bottom = inpt_top.sum(axis=0)

        p_inpt_nb_cond = inpt_top / inpt_bottom
        # If we don't add a small term, the logarithm will make NaNs.
        p_inpt_nb_cond = np.maximum(1E-12, p_inpt_nb_cond)
        entropy = -(p_inpt_nb_cond * np.log(p_inpt_nb_cond)).sum(axis=0)

        diff = entropy - target_entropy
        for j in range(N):
            if abs(diff[j]) < 1e-5:
                continue
            elif diff[j] > 0:
                if maximums[j] == -np.inf or maximums[j] == np.inf:
                    precisions[j] *= 2
                else:
                    precisions[j] = (precisions[j] + maximums[j]) / 2
                minimums[j] = precisions[j]
            else:
                if minimums[j] == -np.inf or minimums[j] == np.inf:
                    precisions[j] /= 2
                else:
                    precisions[j] = (precisions[j] + minimums[j]) / 2
                maximums[j] = precisions[j]

    # Calculcate p matrix once more and return it.
    inpt_top = np.exp(-dists * precisions)
    inpt_top[range(N), range(N)] = 0
    inpt_bottom = inpt_top.sum(axis=0)
    # Add a small constant to the denominator against numerical issues.
    p_inpt_nb_cond = inpt_top / (inpt_bottom + 1e-4)

    # Symmetrize.
    p_ji = (p_inpt_nb_cond + p_inpt_nb_cond.T)

    # We don't normalize correctly here. But that does not matter, since we
    # normalize q wrongly in the same way.
    p_ji /= p_ji.sum()
    p_ji = np.maximum(1E-12, p_ji)

    return p_ji


def build_loss(embeddings):
    """Return a pair (loss, p) given a theano shared variable representing the
    `embeddings`.

    `loss` is a theano variable for the loss. `p` is a symbolic variable
    representing the target neighbour probabilities on which the loss depends.
    """
    # Probability that two points are neighbours in the embedding space.
    emb_dists = distance_matrix(embeddings)
    emb_top = zero_diagonal(1 / (1 + emb_dists))
    emb_bottom = emb_top.sum(axis=0)
    q = emb_top / emb_bottom

    # Incorrect normalization which does not matter since we normalize p i
    # the same way.
    q /= q.sum()
    q = T.maximum(q, 1E-12)

    p_ji_var = T.matrix('neighbour_probabilities')
    p_ji_var_floored = T.maximum(p_ji_var, 1E-12)

    # t-distributed stochastic neighbourhood embedding loss.
    loss = (p_ji_var * T.log(p_ji_var_floored / q)).sum()

    return loss, p_ji_var


class TsneMinimizer(Minimizer):
    """Custom Minimizer for TSNE using a learning rate schedule."""

    def __init__(self, wrt, fprime, steprate, momentum, min_gain=1E-2,
                 args=None):
        super(TsneMinimizer, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.steprate = steprate
        self.momentum = momentum
        self.min_gain = min_gain

    def __iter__(self):
        step_m1 = np.zeros(self.wrt.shape[0])
        gain = np.ones(self.wrt.shape[0])

        for i, (args, kwargs) in enumerate(self.args):
            gradient = self.fprime(self.wrt, *args, **kwargs)
            gain = (gain + 0.2) * ((gradient > 0) != (step_m1 > 0))
            gain += 0.8 * ((gradient > 0) == (step_m1 > 0))
            gain[gain < self.min_gain] = self.min_gain
            step = self.momentum * step_m1
            step -= self.steprate * gradient * gain
            self.wrt += step
            step_m1 = step
            self.wrt -= self.wrt.mean(axis=0)
            yield dict(gradient=gradient, gain=gain, args=args, kwargs=kwargs,
                       n_iter=i, step=step)


def tsne(X, low_dim, perplexity=40, early_exaggeration=50, max_iter=1000,
         verbose=False):
    """Return low dimensional representations for the given data set.

    """
    if early_exaggeration < 0:
        raise ValueError("early_exaggeration has to be non negative")
    if max_iter < 0:
        raise ValueError("max_iter has to be non negative")

    # Define embeddings shared variable and initialize randomly.
    embeddings_flat = theano.shared(np.random.normal(0, 1, X.shape[0] * low_dim))
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
    p_ji_exaggerated = np.maximum(p_ji_exaggerated, 1E-12)

    ee_args = [p_ji_exaggerated] * early_exaggeration
    no_ee_args = itertools.repeat(p_ji)
    args = (([i], {}) for i in itertools.chain(ee_args, no_ee_args))

    opt = TsneMinimizer(embeddings_data, f_d_loss, args=args, momentum=0.5,
                        steprate=500, min_gain=0.01)
    for i, info in enumerate(opt):
        opt.momentum = 0.5 if i < 20 else 0.8
        if verbose:
            print 'loss #%i' % i, f_loss(embeddings_data, p_ji)
            if i == early_exaggeration - 1:
                print 'stopping early exaggeration'
        if i + 1 == max_iter:
            break

    return embeddings_data.reshape(X.shape[0], low_dim)


class Tsne(object):

    def __init__(self, n_inpt, n_lowdim, perplexity=40, early_exaggeration=50,
                 max_iter=1000, verbose=False):
        """Create a Tsne object.

        :param n_inpt: Input dimensionality.
        :param n_lowdim: Desired dimensionality of the representations,
            typically 2 or 3.
        :param perplexity: Parameter to indicate how many `neighbours` a point
            approximately has.
        :param early_exaggeration: Hyper parameter to tune optimization.
        :param max_iter: Number of iterations to perform.
        :param verbose: Flag that indicates whether to print out information during
            the optimization.
        """
        self.n_inpt = n_inpt
        self.n_lowdim = n_lowdim
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.max_iter = max_iter
        self.verbose = verbose

    def fit_transform(self, X):
        """Fit embeddings for `X` and return them.

        :param X: (N, d) shaped array where N is the number of samples and d is
            the dimensionality.
        :returns: (N, n_lowdim) shape array with low dimensional
            representations, where `n_lowdim` has been specified during
            construction.
        """
        return tsne(X, self.n_lowdim, self.perplexity, self.early_exaggeration,
                    self.max_iter, self.verbose)
