# -*- coding: utf-8 -*-


import numpy as np


def roughly(x1, x2, eps=1E-8):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return (abs(x1 - x2) < eps).all()
