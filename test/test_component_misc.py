# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from breze.component.misc import distance_matrix
from tools import roughly


def test_distance_matrix():
    X = T.matrix()
    D = distance_matrix(X)
    f = theano.function([X], D, mode='FAST_COMPILE')
    x = np.array([[1], [2], [3]])
    res = f(x)
    print res
    correct = roughly(res, np.array([[0, 1, 4], [1, 0, 1], [4, 1, 0]]))
    assert correct, 'distance matrix not working right'


