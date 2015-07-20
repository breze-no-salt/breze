# -*- coding: utf-8 -*-


import numpy as np

from nose.tools import with_setup
from theano import tensor as T

from breze.arch.construct.simple import Conv2d
from breze.utils.testhelpers import use_test_values


@with_setup(*use_test_values('raise'))
def test_conv2d_sizes():
    inpt = T.tensor4()

    image_height = 29
    image_width = 13
    inpt.tag.test_value = np.empty((1, 1, image_height, image_width))

    l = Conv2d(inpt, image_height, image_width, 1, 3, 3, 2, subsample=(1, 1))
    calced = [l.output_height, l.output_width]
    real = list(l.output.tag.test_value.shape)[2:]
    msg = 'output shape not calculated right: calced: %s real:%s' % (
        calced, real)
    assert calced == real, msg

    l = Conv2d(inpt, image_height, image_width, 1, 3, 3, 2, subsample=(3, 3))
    calced = [l.output_height, l.output_width]
    real = list(l.output.tag.test_value.shape)[2:]
    msg = 'output shape not calculated right: calced: %s real:%s' % (
        calced, real)
    assert calced == real, msg

    l = Conv2d(inpt, image_height, image_width, 1, 3, 3, 2, subsample=(2, 2))
    calced = [l.output_height, l.output_width]
    real = list(l.output.tag.test_value.shape)[2:]
    msg = 'output shape not calculated right: calced: %s real:%s' % (
        calced, real)
    assert calced == real, msg

    l = Conv2d(inpt, image_height, image_width, 1, 3, 4, 2, subsample=(2, 1))
    calced = [l.output_height, l.output_width]
    real = list(l.output.tag.test_value.shape)[2:]
    msg = 'output shape not calculated right: calced: %s real:%s' % (
        calced, real)
    assert calced == real, msg


@with_setup(*use_test_values('raise'))
def test_conv2d_channels():
    inpt = T.tensor4()

    image_height = 29
    image_width = 13
    n_channels = 2
    inpt.tag.test_value = np.empty((1, n_channels, image_height, image_width))

    Conv2d(inpt, image_height, image_width, 2, 3, 3, 2)
