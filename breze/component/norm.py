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
