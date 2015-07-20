# -*- coding: utf-8 -*-


import itertools
import warnings

import climin.util
import numpy as np
import scipy.linalg
import theano
import theano.tensor as T

try:
    import pywt
except ImportError:
    warnings.warn('pywt not available--limited functionality',
                  ImportWarning)


def window_aggr_filter(func):
    def this_filter(X, window_size):
        filtered = np.empty(X.shape)
        for i in range(X.shape[0]):
            filtered[i] = func(X[max(0, i - window_size):i + 1], axis=0)
        return filtered
    return this_filter


max_filter = window_aggr_filter(np.max)
mean_filter = window_aggr_filter(np.mean)
median_filter = window_aggr_filter(np.median)


def mean_max_filter(X, window_size, decay):
    X = max_filter(X, window_size)
    filtered = np.zeros(X.shape)
    for i in range(1, X.shape[0]):
        filtered[i] = decay * filtered[i - 1] + (1 - decay) * X[i]
    return filtered


# This is a filter which optimizes a filtered signal by trading off
# total variation and the Euclidean distance to the unfiltered signal.
# Since we make use of Theano and do not want to clutter the name space,
# we put it into a pseudo closure of a function.

def _make_tv_filter():
    inpt = T.matrix('inpt')
    filtered_flat = T.vector('filtered_flat')
    filtered = filtered_flat.reshape(inpt.shape)
    diff = filtered[1:] - filtered[:-1]
    total_variation = T.sqrt(diff**2 + 1e-4)
    c_tv = T.scalar('c_tv')
    loss = ((inpt - filtered)**2).mean() + c_tv * total_variation.mean()
    d_loss_wrt_filtered = T.grad(loss, filtered_flat)

    f_loss = theano.function([filtered_flat, inpt, c_tv], loss)
    f_d_loss_wrt_filtered = theano.function([filtered_flat, inpt, c_tv], d_loss_wrt_filtered)

    def _tv_filter(X, c_tv, max_iter=50):
        filtered = X.copy()
        args = itertools.repeat(
            ([X, c_tv], {}))
        opt = climin.util.optimizer('lbfgs', filtered.ravel(), f=f_loss, fprime=f_d_loss_wrt_filtered, args=args)
        for i, info in enumerate(opt):
            if i == max_iter:
                break

        return filtered

    return _tv_filter



tv_filter = _make_tv_filter()


def wavelet_packet_rec(X, wavelet, max_level=3):
    """Return the node wise reconstructions of a full wavelet decompositions.

    :param X: Array of size (n, d), where `d` is the number of different signals;
        each will be decomposed separately.
    :param wavelet: Wavelet to use; string will be passed on to PyWavelets.
    :param max_level: Maximum depth of the resulting packet tree.
    :returns: Array of size `(n, d * 2**3)`.
    """
    res = []
    for i in range(X.shape[1]):
        x = X[:, i]
        wp = pywt.WaveletPacket(x, wavelet, maxlevel=max_level)
        for node in wp.get_leaf_nodes(True):
            res.append(pywt.upcoef(node.path[-1], node.data, wavelet, node.level))
    return np.concatenate([i[:, np.newaxis] for i in res], axis=1)


def wavelet_packet_coef(X, wavelet, max_level=3):
    """Return the node wise coefficients of a full wavelet decompositions.

    :param X: Array of size (n, d), where `d` is the number of different signals;
        each will be decomposed separately.
    :param wavelet: Wavelet to use; string will be passed on to PyWavelets.
    :param max_level: Maximum depth of the resulting packet tree.
    :returns: Array of size `(n / 2**max_level , d * 2**3)`.
    """
    res = []
    for i in range(X.shape[1]):
        x = X[:, i]
        wp = pywt.WaveletPacket(x, wavelet, maxlevel=max_level)
        for node in wp.get_leaf_nodes(True):
            res.append(node.data)

    return np.concatenate([i[:, np.newaxis] for i in res], axis=1)


def savitzky_golay_filter(X, order, degree):
    # Calculate vandermonde matrix.
    rng = np.arange(-order, order + 1, dtype='float64')
    s = np.vander(rng)[:, ::-1]
    S = s[:, :degree + 1]
    r, = scipy.linalg.qr(S, mode='r')[:degree + 1]
    inv_r = scipy.linalg.pinv(r)
    g = np.dot(S, np.dot(inv_r, inv_r.T)).astype('float64')

    filtered = np.zeros_like(X)

    for i in range(1, X.shape[0]):
        if i < order:
            window = np.zeros((order, X.shape[1]))
            window[-i:] = X[:i]
        else:
            window = X[i - order:i]

        filtered[i] = np.dot(g[:order, 0].T, window) * 2

    return filtered


def sg_max_filter(X, window_size, order, degree):
    return savitzky_golay_filter(max_filter(X, window_size), order, degree)
