# coding: utf-8 -*-

import numpy as np
    
from base import roughly

from brummlearn.mda import mda


def test_mda():
    X = np.eye(2)
    w, b = mda(X, 0.5)
    assert roughly(w, [[0.499995, -0.499985], [-0.499985, 0.499995]])
    assert roughly(b, [0.499995, 0.499995])

