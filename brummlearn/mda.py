#!/usr/bin/env python
"""Code for the marginalizing auto encoder."""

__authors__ = ('Christian Osendorfer, osendorf@gmail.com',
               'Justin Bayer, bayer.justin@gmail.com')

import numpy as np
import scipy.linalg as la


from scipy.linalg import lstsq


def mda(X, noise):
    """ Return an affine transformation (w, b) which works as a marginalizing
    denoising autoencoder as described in

        "Marginalized Denoising Autoencoders for Domain Adaptation", Chen et al.

    :param X: array where each row holds one data item.
    :param noise: float that gives the probability that an entry in a data item
        is switched off.
    :returns: Pair (weights, bias) where `weights` is a square matrix `NxN` and
        `bias` is an `N`-dimensional vector, where `N` is the size of a single
        data item.
    """
    # TODO: find better names for q and Q

    # Add another feature for the bias.
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    n, d = X.shape

    q = np.ones((1, d)) * (1 - noise)
    q[0, 0] = 1

    scatter = np.dot(X.T, X)
    Q = scatter * np.dot(q.T, q)
    np.fill_diagonal(Q, q * np.diag(scatter))
    P = scatter * q
    # First row of wm has bias values.
    wm = lstsq((Q + 1e-5 * np.eye(d)).T, P.T[:, 1:])[0]
    return wm[1:], wm[0]
