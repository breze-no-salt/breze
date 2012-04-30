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

    # Normalize.
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

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
        [[  6.95191123e+01],
         [ -5.67286442e-01],
         [  5.99855419e+01],
         [  1.17068631e-02],
         [ -3.95323189e+01]])
    assert roughly(w, desired), 'base not recovered'
