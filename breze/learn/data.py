"""Module for manipulating data."""

import math
import random

import numpy as np
import scipy.interpolate

from sklearn.utils import check_random_state

# TODO numpy doc


def one_hot(array, n_classes=None):
    """Return one of k vectors for an array of class indices.

    :param array: 1D array containing N integers from 0 to k-1.
    :param classes: Amount of classes, k. If None, this will be inferred
        from array (and might take longer).
    :returns: A 2D array of shape (N, k).
    """
    if n_classes is None:
        n_classes = len(set(array.tolist()))
    n = array.shape[0]
    arr = np.zeros((n, n_classes), dtype=np.float32)
    arr[xrange(n), array] = 1.
    return arr


def shuffle(data):
    """Shuffle the first dimension of an indexable object in place."""
    for i in range(len(data) - 2):
        swappartner = random.randint(i + 1, len(data) - 1)
        a, b = data[swappartner][:], data[i][:]
        data[i][:], data[swappartner][:] = b, a


def shuffle_many(arrays, axes, random_state=None):
    rng = check_random_state(random_state)

    # We need to swap the axes of the arrays so that the axes along to shuffle
    # is the first for each. We don't need to swap back, since these will be
    # views.
    arrays = [i.swapaxes(0, j) for i, j in zip(arrays, axes)]

    assert all(i.shape[0] == arrays[0].shape[0] for i in arrays[1:])

    permutation = rng.permutation(arrays[0].shape[0])

    for old_index, new_index in enumerate(permutation):
        for a in arrays:
            a[old_index], a[new_index] = a[new_index], a[old_index]


def padzeros(lst, front=True, return_mask=False):
    # TODO add docs for ``front``
    """Given a list of arrays, pad every array with up front  zeros until they
    reach unit length.

    Each element of `lst` can have a different first dimension, but has to be
    equal on the other dimensions.
    """
    n_items = len(lst)
    # Get the longest item.
    maxlength = max(len(i) for i in lst)
    restshape = list(lst[0].shape)[1:]
    item_shape = [maxlength] + restshape
    total_shape = [n_items] + item_shape

    data = scipy.zeros(total_shape, dtype=lst[0].dtype)
    if return_mask:
        mask = scipy.zeros(total_shape, dtype=lst[0].dtype)
    for i in range(n_items):
        # Iterate over indices because we work in place of the list.
        thislength = lst[i].shape[0]
        if front:
            data[i][-thislength:] = lst[i]
            if return_mask:
                mask[i][-thislength:] = 1
        else:
            data[i][:thislength] = lst[i]
            if return_mask:
                mask[i][:thislength] = 1

    if return_mask:
        return data, scipy.asarray(mask)
    return data


def collapse_seq_borders(arr):
    """Given an array of ndim 3, return a view of ndim 2 where the first
    dimension is flattened out."""
    if not arr.ndim == 3:
        raise ValueError("only for arrays of ndim 3")
    arr2 = arr[:]
    arr2.shape = arr.shape[0] * arr.shape[1], arr.shape[2]
    return arr2


def uncollapse_seq_borders(arr, shape):
    """Return a view of ndim 3, given an array of ndim 2, where the first
    dimension is expanded to 2 dimensions of the given shape."""
    if not arr.ndim == 2:
        raise ValueError("only for arrays of ndim 2")
    arr2 = arr[:]
    arr2.shape = shape[0], shape[1], arr.shape[1]
    return arr2


def skip(X, n, d=1):
    """Return an array X with the same number of rows, but only each `n`'th
    block of `d` consecutive columns is kept.

    Crude way of reducing the dimensionality of time series."""
    X_ = X.reshape((X.shape[0], X.shape[1] / d, d))
    X_ = X_[:, ::n, :]
    return X_.reshape((X.shape[0], X_.size / X.shape[0]))


def interleave(lst):
    """Given a list of arrays, interleave the arrays in a way that the
    first dimension represents the first dimension of every array.

    This is useful for time series, where multiple time series should be
    processed in a single swipe."""
    arr = scipy.asarray(lst)
    return scipy.swapaxes(arr, 0, 1)


def uninterleave(lst):
    """Given an array of interleaved arrays, return an uninterleaved version of
    it."""
    arr = scipy.asarray(lst)
    return scipy.swapaxes(arr, 0, 1)


def interpolate(X, n_intermediates, kind='linear'):
    """Given an array of shape (j, k), return an array of size
    (j * n_intermediates, k) where each i * n_intermediated element refers to
    the i'th element in X while all the others are linearly interpolated."""
    X_ = scipy.empty((X.shape[0] * n_intermediates, X.shape[1]))
    grid = scipy.mgrid[0:X.shape[0]]

    finegrid = scipy.linspace(0, X.shape[0] - 1, X.shape[0] * n_intermediates)
    for i in range(X.shape[1]):
        f = scipy.interpolate.interp1d(grid, X[:, i], kind=kind)
        X_[:, i] = f(finegrid)
    return X_


def n_windows(X, size, offset):
    """Given an array `X` representing a sequence along its first axis, return
    the number of windows of `size` with `offset` that fit into it."""
    return int(math.ceil((X.shape[0] - size + 1.) / offset))


def windowify(X, size, offset=1):
    """Return a static array that represents a sliding window dataset of size
    `size` given by the list of arrays `."""
    # Calculate the amount of windows that fit into one array.
    n_items = sum(n_windows(i, size, offset) for i in X)
    dim = X[0].shape[1]
    X_ = scipy.empty((n_items, size, dim))
    for i, window in enumerate(iter_windows(X, size, offset)):
        X_[i] = window

    return X_


def iter_windows(X, size, offset=1):
    """Return an iterator that goes over a sequential dataset with a sliding
    time window.

    `X` is expected to be a list of arrays, where each array represents a
    sequence along its first axis."""
    for seq in X:
        for j in [k * offset for k in range(n_windows(seq, size, offset))]:
            yield seq[j:j + size]


def split(X, maxlength):
    """Return a list of sequences where each sequence has a length of at most
    `maxlength`.

    Given a list of sequences `X`, the sequences are split accordingly."""
    new_X = []
    for seq in X:
        n_new_seqs, rest = divmod(seq.shape[0], maxlength)
        if rest:
            n_new_seqs += 1
        for i in range(n_new_seqs):
            new_X.append(seq[i * maxlength:(i + 1) * maxlength])
    return new_X


def collapse(X, n):
    """Return a list of sequences, where `n` consecutive timesteps have been
    collapsed into a single timestep by concatenation for each sequence.

    Timesteps are cut off to ensure divisibility by `n`."""
    dim = X[0].shape[1]
    new_X = []
    for seq in X:
        seq = scipy.ascontiguousarray(seq)
        length = seq.shape[0]
        keep = length / n * n
        seq = seq[:keep]
        seq.shape = length / n, dim * n
        new_X.append(seq)

    return new_X


def uncollapse(X, n):
    """Return a list of sequences, where each timestep is divided into `n`
    consecutive timesteps."""
    new_X = []
    dim = X[0].shape[1]
    for seq in X:
        length = seq.shape[0]
        new_X.append(seq.reshape((length * n, dim / n)))
    return new_X


def consecutify(seqs):
    """Given sequences of equal second dimension, put them into a consecutive
    memory block M and return it. Also return a list of views to that block that
    represent the given sequences."""
    n_rows = sum(i.shape[0] for i in seqs)
    n = seqs[0].shape[1]
    block = scipy.empty((n_rows, n))
    new_seqs = []
    start = 0
    for seq in seqs:
        stop = start + seq.shape[0]
        block[start:stop] = seq
        new_seqs.append(block[start:stop])
        start = stop
    return block, new_seqs


def sample(arr, n, axis=0, with_replacement=False):
    indices = range(arr.shape[axis])
    all_slice = slice(0, arr.shape[axis])
    sampled_indices = random.sample(indices, n)
    slices = tuple(all_slice if i != axis else sampled_indices
                   for i in range(arr.ndim))
    return arr[slices]
