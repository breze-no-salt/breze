# -*- coding: utf-8 -*-


import theano.tensor as T


def l1(inpt, axis=None):
    return abs(inpt).sum(axis=axis)


def l2(inpt, axis=None):
    return (inpt**2).sum(axis=axis)


def root_l2(inpt, axis=None):
    return T.sqrt((inpt**2).sum(axis=axis))


def exp(inpt, axis=None):
    return T.exp(inpt).sum(axis=axis)


def normalize(inpt, f_comp, axis, eps=1E-8):
    if axis not in (0, 1):
        raise ValueError('only axis 0 or 1 allowed')

    transformed = f_comp(inpt)
    this_norm = transformed.sum(axis=axis)
    if axis == 0:
        res = transformed / (this_norm + eps)
    elif axis == 1:
        res = (transformed.T / (this_norm + eps)).T

    return res
