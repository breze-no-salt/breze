#!/usr/bin/env python

import numpy as np

from brummlearn.tsne import tsne


def test_tsne():
    X = np.random.random((100, 3))
    E = tsne(X, 2, 40, 10, 30)
