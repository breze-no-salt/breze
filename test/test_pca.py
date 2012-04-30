# coding: utf-8 -*-

import numpy as np
import scipy.linalg
    
from base import roughly

from brummlearn.pca import pca


def test_pca():
    l = np.arange(0,  1, 0.001).reshape((1000, 1))
    expand = np.array([2.5, 1.2]).reshape((1, 2))

    X = np.dot(l, expand)
    X += np.random.normal(0, 1E-6, X.shape) 

    w, s  = pca(X, 1)

    desired = np.array([[-0.9015], [-0.4327]])
    print s

    assert roughly(w, desired, 1E-3)


def test_pca():
    l = np.arange(0,  1, 0.001).reshape((1000, 1))
    expand = np.array([2.5, 1.2]).reshape((1, 2))

    X = np.dot(l, expand)
    X += np.random.normal(0, 1E-6, X.shape) 

    w, s  = pca(X, 2, whiten=True)

    X = np.dot(X, w)
    assert roughly(np.cov(X, rowvar=0), np.eye(2), 1E-2), 'covariance not white'
