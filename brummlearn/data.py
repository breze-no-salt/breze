"""Module for manipulating data."""

import itertools
import math
import random

import scipy.interpolate


def shuffle(data):
    """Shuffle the first dimension of an indexable object in place."""
    for i in range(len(data) - 2):
        swappartner = random.randint(i + 1, len(data) - 1)
        a, b = data[swappartner][:], data[i][:]
        data[i][:], data[swappartner][:] = b, a


def padzeros(lst):
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
    
    data = scipy.zeros(total_shape)
    for i in range(n_items):
        thislength = lst[i].shape[0]
        data[i][-thislength:] = lst[i]

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

    
def minibatches(arr, batchsize, d=0):
    """Return a list of views of the given arr.

    Given `batchsize`, each batch but the last one will have the specified
    size. Slicing will be performed along the `d`th dimension."""
    n_batches, rest = divmod(arr.shape[d], batchsize)
    if rest != 0:
        n_batches += 1
        
    slices = (slice(i * batchsize, (i + 1) * batchsize)
              for i in range(n_batches))
    if d == 0:
        res = [arr[i] for i in slices]
    elif d == 1:
        res = [arr[:, i] for i in slices]
    elif d == 2:
        res = [arr[:, :, i] for i in slices]

    return res


def iter_minibatches(lst, batchsize, dims):
    """Return an iterator that successively yields tuples containing aligned
    minibatches of size `batchsize` from slicable objects given in `lst`.

    Because different containers might require slicing over different
    dimensions, the dimension of each container has to be givens as a list
    `dims`."""
    batches = [minibatches(i, batchsize, d) for i, d in zip(lst, dims)]
    if len(batches) > 1:
        if any(len(i) != len(batches[0]) for i in batches[1:]):
            raise ValueError("containers to be batched have different lengths")
    while True:
        indices = [i for i, _ in enumerate(batches[0])]
        while True:
            random.shuffle(indices)
            for i in indices:
                yield tuple(b[i] for b in batches)


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
    x_count = itertools.count(0)
    for seq in X:
        for j in [k * offset for k in range(n_windows(seq, size, offset))]:
            i = x_count.next()
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
