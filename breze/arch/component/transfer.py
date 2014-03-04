# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
import theano.tensor.nnet
import norm

# TODO documentall of these


def tanh(inpt):
    return T.tanh(inpt)


def tanhplus(inpt):
    return T.tanh(inpt) + inpt


def sigmoid(inpt):
    return T.nnet.sigmoid(inpt)


def rectifier(inpt):
    return T.maximum(inpt, 0)


def softplus(inpt):
    return T.log(1 + T.exp(inpt))


def identity(inpt):
    return inpt


def logproduct_of_t(inpt):
    return T.log(1 + inpt**2)


softmax = T.nnet.softmax


def logcosh(inpt):
    return T.log(T.cosh(inpt))


def softabs(inpt, eps=1E-5):
    return T.sqrt(inpt**2 + eps)


def softsign(inpt):
    return inpt / (1 + abs(inpt))
