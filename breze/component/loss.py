# -*- coding: utf-8 -*-

"""Module containing several losses usable for supervised and unsupervised
training.

The only restriction that is imposed on losses in general is that they
take Theano variables as arguments and return Theano variables as arguments
as well.

The API is agnostic about the exact definition how those arguments come
to pass. Here are some abstrations.


- _Supervised pairwise losses_ take two arguments, the target and the given
  prediction of a model (in that order) of the same shape. The elements of
  the two variables are paired coordinate-wise. The return value is of the
  same shape and contains the loss that results from the respective pairs
  at the corresponding coordinates.

"""


import numpy as np
import theano.tensor as T

from misc import distance_matrix


def squared(target, prediction):
    """Return the element wise squared loss between the `target` and
    the `prediction`.

    :param target: An array of arbitrary shape representing representing
        the targets.
    :param prediction: An array of the same shape as `target`.
    :returns: An array of the same size as `target` and `prediction`xi
        representing the pairwise divergences."""
    return (target - prediction) ** 2


def absolute(target, prediction):
    """Return the element wise absolute difference between the `target` and
    the `prediction`.

    :param target: An array of arbitrary shape representing the targets.
    :param prediction: An array of the same shape as `target`.
    :returns: An array of the same size as `target` and `prediction`xi
        representing the pairwise divergences."""
    return abs(target - prediction)


def nce(target, prediction):
    """Return the element wise negative cross entropy between the `target` and
    the `prediction`.

    Used for classification purposes.

    The loss is different to `nnce` by that `prediction` is not an array of
    integers but a hot k coding.

    :param target: An array of shape `(n, k)` where `n` is the number of
        samples and k is the number of classes. Each row represents a hot k
        coding. It should be zero except for one element, which has to be
        exactly one.
    :param prediction: An array of shape `(n, k)`. Each row is
        interpreted as a categorical probability. Thus, each row has to sum
        up to one and be strictly positive for this measure to make sense.
    :returns: An array of the same size as `target` and `prediction`
        representing the pairwise divergences."""
    return -(target * T.log(prediction))


def nnce(target, prediction):
    """Return the element wise negative cross entropy between the `target` and
    the `prediction`.

    Used for classification purposes.

    The loss is different to `nce` by that `prediction` is not a hot k coding
    but an array of integers.

    :param target: An array of shape `(n,)` where `n` is the number of
        samples. Each entry of the array should be an integer between `0` and
        `k-1`, where `k` is the number of classes.
    :param prediction: An array of shape `(n, k)`. Each row is
        interpreted as a categorical probability. Thus, each row has to sum
        up to one and be strictly positive for this measure to make sense.
    :returns: An array of shape `(n, 1)` as `target` containing the log
        probability that that example is classified correctly."""
    loss_vector = -(T.log(prediction)[T.arange(target.shape[0]), target])
    # To be compatible with the API, we make this a (n, 1) matrix.
    return T.shape_padright(loss_vector)


def nces(target, prediction):
    """Return the negative cross entropies between binary vectors (the targets)
    and a number of Bernoulli variables (the predictions).

    Used in regression on binary variables, not classification.

    :param target: An array of shape `(n, k)` where `n` is the number of
        samples and k is the number of outputs. Each entry should be either 0 or
        1.
    :param prediction: An array of shape `(n, k)`. Each row is
        interpreted as a set of statistics of Bernoulli variables. Thus, each
        element has to lie in (0, 1).
    :returns: An array of the same size as `target` and `prediction`
        representing the pairwise divergences."""
    return -(target * T.log(prediction) + (1 - target) * T.log(1 - prediction))


def bern_bern_kl(X, Y, axis=None):
    """Return the Kullback-Leibler divergence between Bernoulli variables
    represented by their sufficient statistics.

    :param X: An array of arbitrary shape where each element represents
        the statistic of a Bernoulli variable and thus should lie in (0, 1).
    :param Y: An array of the same shape as `target` where each element
        represents the statistic of a Bernoulli variable and thus should lie in
        (0, 1).
    :returns: An array of the same size as `target` and `prediction`
        representing the pairwise divergences."""
    return X * T.log(X / Y) + (1 - X) * T.log((1 - X) / (1 - Y))


def ncac(target, embedding):
    """Return the sample wise NCA for classification method.

    This corresponds to the probability that a point is correctly classified
    with a soft knn classifier using leave-one-out. Each neighbour is weighted
    according to an exponential of its negative Euclidean distance. Afterwards,
    a probability is calculated for each class depending on the weights of the
    neighbours. For details, we refer you to

    'Neighbourhood Component Analysis' by
    J Goldberger, S Roweis, G Hinton, R Salakhutdinov (2004).

    :param target: An array of shape `(n,)` where `n` is the number of
        samples. Each entry of the array should be an integer between `0` and
        `k-1`, where `k` is the number of classes.
    :param embedding: An array of shape `(n, d)` where each row represents
        a point in d dimensional space.
    :returns: Array of shape `(n, 1)`.
    """
    # Matrix of the distances of points.
    dist = distance_matrix(embedding)
    thisid = T.identity_like(dist)

    # Probability that a point is neighbour of another point based on
    # the distances.
    top = T.exp(-dist) + 1e-8       # Add a small constant for stability.
    bottom = (top - thisid * top).sum(axis=0)
    p = top / bottom

    # Create a matrix that matches same classes.
    sameclass = T.eq(distance_matrix(target), 0) - thisid
    loss_vector = -(p * sameclass).sum(axis=1)
    # To be compatible with the API, we make this a (n, 1) matrix.
    return T.shape_padright(loss_vector)


def ncar(target, embedding):
    """Return the sample wise NCA for regression loss.

    This is similar to NCA for classification, except that not soft KNN
    classification but regression performance is maximized. (Actually, the
    negative performance is minimized.)

    For details, we refer you to

    'Pose-sensitive embedding by nonlinear nca regression' by
    Taylor, G. and Fergus, R. and Williams, G. and Spiro, I. and Bregler, C.
    (2010)

    :param target: An array of shape `(n, d)` where `n` is the number of
        samples and `d` the dimensionalty of the target space.
    :param embedding: An array of shape `(n, d)` where each row represents
        a point in d dimensional space.
    :returns: Array of shape `(n, 1)`.
    """
    # Matrix of the distances of points.
    dist = distance_matrix(embedding)
    thisid = T.identity_like(dist)

    # Probability that a point is neighbour of another point based on
    # the distances.
    top = T.exp(-dist) + 1E-8  # Add a small constant for stability.
    bottom = (top - thisid * top).sum(axis=0)
    p = top / bottom

    # Create matrix of distances.
    target_distance = distance_matrix(target, target, 'soft_l1')
    # Set diagonal to 0.
    target_distance -= target_distance * T.identity_like(target_distance)

    loss_vector = (p * target_distance ** 2).sum(axis=1)
    # To be compatible with the API, we make this a (n, 1) matrix.
    return T.shape_padright(loss_vector)


def drlim(push_margin, pull_margin, c_contrastive):
    """Return a function that implements the

    'Dimensionality reduction by learning an invariant mapping' by
    Hadsell, R. and Chopra, S. and LeCun, Y. (2006).

    For an example of such a function, see `drlim1` with a margin of 1.
    
    Parameters
    ----------
    
    push_margin : Float
        The minimum margin that negative pairs should be seperated by.
        Pairs seperated by higher distance than push_margin will not
        contribute to the loss.

    pull_margin: Float
        The maximum margin that positive pairs may be seperated by.
        Pairs seperated by lower distances do not contribute to the loss.
    
    c_contrastive : Float
        Coefficient to weigh the contrastive term relative to the 
        positive term

    Returns
    -------

    loss : callable
        Function that takes two arguments, a target and an embedding."""

    def inner(target, embedding):
        """Return a theano expression of a vector containing the sample wise
        loss of drlim.

        The push_margin, pull_margin and coefficient for the contrastives 
        used are %.f, %.f and %.f respectively.

        Parameters
        ----------

        target : array_like
            A vector of length `n`. If 1, sample `2 * n` and sample 
            `2 * n + 1` are deemed similar.

        embedding : array_like
            Array containing the embeddings of samples row wise.
        """ % (push_margin, pull_margin, c_contrastive)
        target = target[:, 0]
        n_pair = embedding.shape[0] / 2
        n_feature = embedding.shape[1]

        # Reshape array to get pairs.
        embedding = embedding.reshape((n_pair, n_feature * 2))

        # Calculate distances of pairs.
        diff = (embedding[:, :n_feature] - embedding[:, n_feature:])
        dist = T.sqrt((diff ** 2).sum(axis=1) + 1e-8)

        pull = target * T.maximum(0, dist - pull_margin)
        push = (1 - target) * T.maximum(0, push_margin - dist) ** 2

        loss = pull + c_contrastive * push
        return loss.dimshuffle(0, 'x')

    return inner


drlim1 = drlim(1, 0, 0.5)


def diag_gaussian_nll(target, prediction):
    n_output = prediction.shape[-1] / 2
    if prediction.ndim == 3:
        # We have dynamic data.
        mean, std = prediction[:, :, :n_output], prediction[:, :, n_output:]
    elif prediction.ndim == 2:
        # We have static data.
        mean, std = prediction[:, :n_output], prediction[:, n_output:]
    var = (std + 1e-4) ** 2
    residuals = target - mean
    weighted_squares = -(residuals ** 2) / (2 * var + 1e-3)
    normalization = T.log(T.sqrt(2 * np.pi * var + 1e-3))
    ll = weighted_squares - normalization
    return -ll
