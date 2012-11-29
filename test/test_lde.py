# coding: utf-8 -*-

import numpy as np

from base import roughly

from brummlearn.lde import LinearDenoiser


def test_lde():
    X = np.eye(2)
    lde = LinearDenoiser(2, 0.5)
    lde.fit(X)
    assert roughly(lde.weights, [[0.499995, -0.499985], [-0.499985, 0.499995]])
    assert roughly(lde.bias, [0.499995, 0.499995])

