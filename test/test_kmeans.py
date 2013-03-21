# coding: utf-8 -*-

import numpy as np

from brummlearn.kmeans import GainShapeKMeans


def test_gainshapekmeans():
    X = np.random.normal(0, 1E-6, (10, 10))
    kmeans = GainShapeKMeans(3)
    kmeans.fit(X)

    kmeans.transform(X, 'linear')
    Y = kmeans.transform(X, 'omp-1')
    assert np.allclose((Y != 0).sum(axis=1), np.ones_like(Y[:, 0]))
