#!/usr/bin/env python

import numpy as np
import scipy.linalg
    
from base import roughly

from brummlearn.sfa import sfa


def test_sfa():
    t = np.arange(0,  2 * np.pi, 0.01)
    X1 = np.sin(t) + np.cos(11 * t)**2
    X2 = np.cos(11 * t)

    # Feature expansion.
    X = np.vstack([X1, X2, X1 * X2, X1**2, X2**2]).T

    ## Normalize.
    X -= X.mean(axis=0)

    # Whiten.
    n, d = X.shape
    n_comp = d
    cov = np.cov(X, rowvar=0)

    w, s, v = scipy.linalg.svd(cov, full_matrices=False)
    w = w[:, :n_comp]
    
    w = np.dot(w, scipy.diag(1. / scipy.sqrt(s[:n_comp])))
    X = np.dot(X, w)

    w = sfa([X], 1)
    desired = np.array(
        [[ 8.03244781e-01],
         [ -1.23768922e-03],
         [ 2.21901512e-01],
         [ -5.92845153e-04],
         [ 5.52770891e-01]])
    assert roughly(w, desired), 'base not recovered'

