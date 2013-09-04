# coding: utf-8 -*-

import numpy as np

from breze.learn.lde import LinearDenoiser


def test_lde():
    X = np.eye(2)
    lde = LinearDenoiser(0.5)
    lde.fit(X)
    assert np.allclose(lde.weights, [[0.499995, -0.499985], [-0.499985, 0.499995]])
    assert np.allclose(lde.bias, [0.499995, 0.499995])
