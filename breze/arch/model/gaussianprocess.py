# -*- coding: utf-8 -*-

"""Module containing models for Gaussian processes."""


import numpy as np
import theano.tensor as T

from theano.sandbox.linalg.ops import MatrixInverse, Det, psd, Cholesky
minv = MatrixInverse()
det = Det()
cholesky = Cholesky()

from ..util import lookup, get_named_variables
from ..component import misc, kernel as kernel_


def parameters(n_inpt):
    return dict(length_scales=n_inpt, noise=1, amplitude=1)


def exprs(inpt, test_inpt, target, length_scales, noise, amplitude, kernel):
    exprs = {}
    # To stay compatible with the prediction api, target will
    # be a matrix to the outside. But in the following, it's easier
    # if it is a vector in the inside. We keep a reference to the
    # matrix anyway, to return it.

    target_ = target[:, 0]

    # The Kernel parameters are parametrized in the log domain. Here we recover
    # them and make sure that they do not get zero.
    minimal_noise = 1e-4
    minimal_length_scale = 1e-4
    minimal_amplitude = 1e-4

    noise = T.exp(noise) + minimal_noise
    length_scales = T.exp(length_scales) + minimal_length_scale
    amplitude = T.exp(amplitude) + minimal_amplitude

    # In the case of stationary kernels (those which work on the distances
    # only) we can save some work by caching the distances. Thus we first
    # find out if it is a stationary tensor by checking whether the kernel
    # can be computed by looking at diffs only---this is the case if a
    # ``XXX_by_diff`` function is available in the kernel module.
    # If that is the case, we add the diff expr to the exprs dict, so it can
    # be exploited by code on the top via a givens directory.

    kernel_by_dist_func = lookup('%s_by_dist' % kernel, kernel_, None)
    stationary = kernel_by_dist_func is not None
    kernel_func = lookup(kernel, kernel_)

    if stationary:
        inpt_scaled = inpt * length_scales.dimshuffle('x', 0)
        diff = exprs['diff'] = misc.pairwise_diff(inpt_scaled, inpt_scaled)
        D2 = exprs['sqrd_dist'] = misc.distance_matrix_by_diff(diff, 'l2')
        gram_matrix = amplitude * kernel_by_dist_func(D2)
        exprs['D2'] = D2
    else:
        gram_matrix = kernel_func(inpt, inpt, length_scales, amplitude)

    gram_matrix += T.identity_like(gram_matrix) * noise

    # This is an informed choice. I played around a little with various
    # methods (e.g. using cholesky first) and came to the conclusion that
    # this way of doing it was way faster than explicitly doing a Cholesky
    # or so.

    psd(gram_matrix)
    inv_gram_matrix = minv(gram_matrix)

    n_samples = gram_matrix.shape[0]
    ll = (
        - 0.5 * T.dot(T.dot(target_.T, inv_gram_matrix), target_)
        - 0.5 * T.log(det(gram_matrix))
        - 0.5 * n_samples * T.log(2 * np.pi))
    nll = -ll

    # We are interested in a loss that is invariant to the number of
    # samples.
    nll /= n_samples
    loss = nll

    # Whenever we are working with points not in the training set, the
    # corresponding expressions are prefixed with test_. Thus test_inpt,
    # test_K (this is the Kernel matrix of the test inputs only), and
    # test_kernel (this is the kernel matrix of the training inpt with the
    # test inpt.
    test_kernel = kernel_func(inpt, test_inpt, length_scales, amplitude)

    kTK = T.dot(test_kernel.T, inv_gram_matrix)
    output = output_mean = T.dot(kTK, target_).dimshuffle(0, 'x')

    kTKk = T.dot(kTK, test_kernel)

    chol_inv_gram_matrix = cholesky(inv_gram_matrix)

    diag_kTKk = (T.dot(chol_inv_gram_matrix.T, test_kernel) ** 2).sum(axis=0)
    test_K = kernel_func(test_inpt, test_inpt, length_scales, amplitude,
                         diag=True)
    output_var = ((test_K - diag_kTKk)).dimshuffle(0, 'x')

    return get_named_variables(locals())
