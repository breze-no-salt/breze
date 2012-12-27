# -*- coding: utf-8 -*-


import itertools

import climin.util
import numpy as np
import theano
import theano.tensor as T


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
