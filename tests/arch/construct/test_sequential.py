# -*- coding: utf-8 -*-


from theano import tensor as T

from breze.arch.construct.sequential import Recurrent
from breze.arch.util import ParameterSet


def test_transfer_insize_outsize():
    inpt = T.tensor3('inpt')

    def t(x):
        return x
    t.in_size = 2
    t.out_size = 3

    P = ParameterSet()
    r = Recurrent(inpt, 4, t, P.declare)
    P.alloc()

    s = P[r.weights].shape
    assert  s == (12, 8), 'Shape is %s' % str(s)

    s = P[r.initial].shape
    assert s == (12,),  'Shape is %s' % str(s)

