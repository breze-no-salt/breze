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


import theano.tensor as T

from misc import distance_matrix


# TODO use numpydoc
# TODO nce, nnce, nces and nnces are stupid names. They refer to a more general
# concept, the cross entropy, although they really are Bernoulli or Multinomial
# cross entropies. Better names required.
# TODO add hinge loss
# TODO add huber loss


def squared(target, prediction):
    """Return the element wise squared loss between the `target` and
    the `prediction`.

    Parameters
    ----------

    target : Theano variable
        An array of arbitrary shape representing representing the targets.

    prediction : Theano variable
        An array of arbitrary shape representing representing the predictions.

    Returns
    -------

    res : Theano variable
        An array of the same shape as ``target`` and ``prediction``
        representing the pairwise distances."""
    return (target - prediction) ** 2


def absolute(target, prediction):
    """Return the element wise absolute difference between the ``target`` and
    the ``prediction``.

    Parameters
    ----------

    target : Theano variable
        An array of arbitrary shape representing representing the targets.

    prediction : Theano variable
        An array of arbitrary shape representing representing the predictions.

    Returns
    -------

    res : Theano variable
        An array of the same shape as ``target`` and ``prediction``
        representing the pairwise distances."""
    return abs(target - prediction)


def multinomial_ce(target, prediction, eps=1e-8):
    """Return the cross entropy between the ``target`` and the ``prediction``,
    where ``prediction`` is a summary of the statistics of the multinomial and
    ``target`` is a some outcome.

    Used for multiclass classification purposes.

    The loss is different to ``nmultinomial_ce`` by that ``target`` is not
    an array of integers but a hot k coding.

    Note that predictions are clipped between ``eps`` and ``1 - eps`` to ensure
    numerical stability.

    Parameters
    ----------

    target : Theano variable
        An array of shape ``(n, k)`` where ``n`` is the number of samples and
        ``k`` is the number of classes. Each row represents a hot k
        coding. It should be zero except for one element, which has to be
        exactly one.

    prediction : Theano variable
        An array of shape ``(n, k)``. Each row is interpreted as a categorical
        probability. Thus, each row has to sum up to one and be non-negative.

    Returns
    -------

    res : Theano variable.
        An array of the same size as ``target`` and ``prediction`` representing
        the pairwise divergences."""
    prediction = T.clip(prediction, eps, 1 - eps)
    return -(target * T.log(prediction))


def nmultinomial_ce(target, prediction):
    """Return the cross entropy between the ``target`` and the ``prediction``,
    where ``prediction`` is a summary of the statistics of the multinomial and
    ``target`` is a some outcome.

    Used for classification purposes.

    The loss is different to ``multinomial_ce`` by that ``target`` is not a hot
    k coding but an array of integers.

    Parameters
    ----------

    target : Theano variable
        An array of shape ``(n,)`` where `n` is the number of samples. Each
        entry of the array should be an integer between ``0`` and ``k-1``,
        where ``k`` is the number of classes.
    prediction : Theano variable
        An array of shape ``(n, k)`` or ``(t, n , k)``. Each row (i.e. entry in
        the last dimension) is interpreted as a categorical probability. Thus,
        each row has to sum up to one and be non-negative.


    Returns
    -------

    res : Theano variable
        An array of shape ``(n, 1)`` as ``target`` containing the log
        probability that that example is classified correctly."""

    # The following code might seem more complicated as necessary. Yet,
    # at the time of writing the gradient of AdvancedIncSubtensor does not run
    # on the GPU, which is why we reduce it to using AdvancedSubtensor.

    if prediction.ndim == 3:
        # We are looking at a 3D problem (e.g. via recurrent nets) and this
        # make it a 2D problem.
        target_flat = target.flatten()
        prediction_flat = prediction.flatten()
    elif prediction.ndim == 2:
        target_flat = target
        prediction_flat = prediction.flatten()
    else:
        raise ValueError('only 2 or 3 dims supported for nnce')
    target_flat.name = 'target_flat'
    prediction_flat.name = 'prediction_flat'

    target_flat += T.arange(target_flat.shape[0]) * prediction.shape[-1]

    # This cast needs to be explicit, because in the case of the GPU, the
    # targets will always be floats.
    target_flat = T.cast(target_flat, 'int32')
    loss = -T.log(prediction_flat)[target_flat]

    # In both forks below, a trailing 1 is added to the shape because that
    # is what the caller expects. (As it is e.g. with the squared error.)
    if prediction.ndim == 3:
        # Convert back from 2D to 3D.
        loss = loss.reshape((prediction.shape[0], prediction.shape[1], 1))
    elif prediction.ndim == 2:
        loss = loss.reshape((prediction.shape[0], 1))

    return loss


def bernoulli_ces(target, prediction):
    """Return the Bernoulli cross entropies between binary vectors ``target``
    and a number of Bernoulli variables ``prediction``.

    Used in regression on binary variables, not classification.

    Parameters
    ----------

    target : Theano variable
        An array of shape ``(n, k)`` where ``n`` is the number of samples and k
        is the number of outputs. Each entry should be either 0 or 1.

    prediction : Theano variable.
        An array of shape ``(n, k)``. Each row is interpreted as a set of
        statistics of Bernoulli variables. Thus, each element has to lie in
        ``(0, 1)``.

    Returns
    -------

    res : Theano variable
        An array of the same size as ``target`` and ``prediction`` representing
        the pairwise divergences.
    """
    return -(target * T.log(prediction) + (1 - target) * T.log(1 - prediction))


def bern_bern_kl(X, Y):
    """Return the Kullback-Leibler divergence between Bernoulli variables
    represented by their sufficient statistics.

    Parameters
    ----------

    X : Theano variable
        An array of arbitrary shape where each element represents
        the statistic of a Bernoulli variable and thus should lie in
        ``(0, 1)``.
    Y : Theano variable
        An array of the same shape as ``target`` where each element represents
        the statistic of a Bernoulli variable and thus should lie in
        ``(0, 1)``.

    Returns
    -------

     res : Theano variable
        An array of the same size as ``target`` and ``prediction`` representing
        the pairwise divergences."""
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


def drlim(push_margin, pull_margin, c_contrastive,
          push_loss='squared', pull_loss='squared'):
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

    push_loss : One of {'squared', 'absolute'}, optional, default: 'squared'
        Loss to encourage Euclidean distances between non pairs.

    pull_loss : One of {'squared', 'absolute'}, optional, default: 'squared'
        Loss to punish Euclidean distances between pairs.

    Returns
    -------

    loss : callable
        Function that takes two arguments, a target and an embedding."""

    # One might think that we'd need to use abs as the non-squared loss here.
    # Yet, due to the maximum operation later one we can just take the identity
    # as well.
    f_push_loss = T.square if push_loss == 'squared' else lambda x: x
    f_pull_loss = T.square if pull_loss == 'squared' else lambda x: x

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
        n_pair = embedding.shape[0] // 2
        n_feature = embedding.shape[1]

        # Reshape array to get pairs.
        embedding = embedding.reshape((n_pair, n_feature * 2))

        # Calculate distances of pairs.
        diff = (embedding[:, :n_feature] - embedding[:, n_feature:])
        dist = T.sqrt((diff ** 2).sum(axis=1) + 1e-8)

        pull = target * f_pull_loss(T.maximum(0, dist - pull_margin))
        push = (1 - target) * f_push_loss(T.maximum(0, push_margin - dist))

        loss = pull + c_contrastive * push
        return loss.dimshuffle(0, 'x')

    return inner


drlim1 = drlim(1, 0, 0.5)
