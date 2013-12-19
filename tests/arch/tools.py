# -*- coding: utf-8 -*-


import theano
import numpy as np


def test_values_off():
    theano.config.compute_test_value = 'off'


def test_values_raise():
    theano.config.compute_test_value = 'raise'
