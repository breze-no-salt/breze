# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import numpy as np

from breze.arch.component.misc import distance_matrix, project_into_l2_ball
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


def test_project_into_l2_ball_single():
    x = T.vector()
    x_projected = project_into_l2_ball(x, 1)
    f = theano.function([x], x_projected)

    x = np.array([.1, .1, .1])
    y = f(x)
    assert np.allclose(np.array([.1, .1, .1]), y)

    x = np.array([2, 1, 1])
    y = f(x)
    assert np.allclose(np.array([0.81649658, 0.40824829, 0.40824829]), y)

    x = np.array([0, 1, 0])
    y = f(x)
    assert np.allclose(np.array([0., 1, 0]), y)


def test_project_into_l2_ball_batch():
    x = T.matrix()
    x_projected = project_into_l2_ball(x, 1)
    f = theano.function([x], x_projected)

    x = np.array([[.1, .1, .1],
                  [0, 1, 0]])
    y = f(x)

    desired = np.array([
        [.1, .1, .1],
        [0., 1, 0],
    ])

    print y
    assert np.allclose(desired, y)
