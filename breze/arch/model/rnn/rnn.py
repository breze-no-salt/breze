# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
from theano.tensor.extra_ops import repeat


def recurrent_layer(hidden_inpt, hidden_to_hidden, f, initial_hidden):
    def step(x, hi_tm1):
        h_tm1 = f(hi_tm1)
        hi = T.dot(h_tm1, hidden_to_hidden) + x
        return hi

    # Modify the initial hidden state to obtain several copies of
    # it, one per sample.
    # TODO check if this is correct; FD-RNNs do it right.
    initial_hidden_b = repeat(initial_hidden, hidden_inpt.shape[1], axis=0)
    initial_hidden_b = initial_hidden_b.reshape(
        (hidden_inpt.shape[1], hidden_inpt.shape[2]))

    hidden_in_rec, _ = theano.scan(
        step,
        sequences=hidden_inpt,
        outputs_info=[initial_hidden_b])

    hidden_rec = f(hidden_in_rec)

    return hidden_in_rec, hidden_rec


def recurrent_layer_stateful(hidden_inpt, hidden_to_hidden, f, initial_hidden):
    def step(x, s_m1, hi_tm1, h_tm1):
        hi = T.dot(h_tm1, hidden_to_hidden)
        hi += x
        s, h = f(s_m1, hi)
        return s, hi, h

    initial_hidden_b = repeat(
        initial_hidden.dimshuffle('x', 0), hidden_inpt.shape[1], axis=0)

    (states, hidden_in_rec, hidden_rec), _ = theano.scan(
        step,
        sequences=hidden_inpt,
        outputs_info=[
            T.zeros_like(initial_hidden_b),
            T.zeros_like(hidden_inpt[0]),
            initial_hidden_b])

    return states, hidden_in_rec, hidden_rec
