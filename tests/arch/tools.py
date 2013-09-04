# -*- coding: utf-8 -*-


import theano
import numpy as np


def roughly(x1, x2, eps=1E-8):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return (abs(x1 - x2) < eps).all()


def test_values_off():
    theano.config.compute_test_value = 'off'


def test_values_raise():
    theano.config.compute_test_value = 'raise'
