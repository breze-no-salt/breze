# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T
import math

import breze.arch.util
from breze.arch.util import ParameterSet, Model


def test_parameter_set_init():
    pars = ParameterSet(matrix=(10, 10),
                        vector=10)
    assert pars.data.shape == (110,), 'wrong size for flat pars allocated'
    assert (pars['matrix'].shape == (10, 10)), ('wrong size for 2d array in pars '
        'allocated')
    assert (pars['vector'].shape == (10,)), ('wrong size for 1d array in pars '
        'allocated')


def test_parameter_set_data_change():
    pars = ParameterSet(matrix=(10, 10),
                        vector=10)
    pars['matrix'][...] = 0
    pars['vector'][...] = 0
    assert (pars.data == 0).all(), repr(pars.data)

    pars['matrix'] += 1
    assert pars.data.sum() == 100

    pars['vector'] += 2
    assert pars.data.sum() == 120

    pars.data *= 0.5
    pars.data.sum() == 60


def test_model_function():
    pars = ParameterSet(weights=(2, 3))
    inpt = T.matrix()
    output = T.dot(inpt, pars.weights)
    pars.data[...] = np.random.standard_normal(pars.data.shape)

    model = Model()
    model.exprs = {'inpt': inpt, 'output': output}
    model.parameters = pars

    f = model.function(['inpt'], 'output')
    fx = model.function(['inpt'], 'output', explicit_pars=True)

    np.random.seed(1010)
    test_inpt = np.random.random((10, 2)).astype(theano.config.floatX)
    r1 = f(test_inpt)
    r2 = fx(pars.data, test_inpt)
    print r1
    print r2
    correct = np.allclose(r1, r2)

    assert correct, 'implicit pars and explicit pars have different output'

    f1 = model.function(['inpt'], ['output'])
    f2 = model.function([inpt], ['output'])
    f3 = model.function([inpt], [output])
    f4 = model.function(['inpt'], [output])

    assert np.allclose(f1(test_inpt), f2(test_inpt)), "f1 and f2 don't agree"
    assert np.allclose(f1(test_inpt), f3(test_inpt)), "f1 and f3 don't agree"
    assert np.allclose(f1(test_inpt), f4(test_inpt)), "f1 and f4 don't agree"
    assert np.allclose(f2(test_inpt), f3(test_inpt)), "f2 and f3 don't agree"
    assert np.allclose(f2(test_inpt), f4(test_inpt)), "f2 and f4 don't agree"
    assert np.allclose(f3(test_inpt), f4(test_inpt)), "f3 and f4 don't agree"


def test_flatten():
    nested = (1, 2, [3, 4, 5, [6], [7], (8), ([9, 10, []]), (), ((([[]]))), []], 11)
    flattened = breze.arch.util.flatten(nested)
    rev_flattened = flattened[::-1]
    renested = breze.arch.util.unflatten(nested, flattened)
    rev_renested = breze.arch.util.unflatten(nested, rev_flattened)
    print "nested:              ", nested
    print "flattened:           ", flattened
    print "renested:            ", renested
    print "reversed flattened:  ", rev_flattened
    print "reversed renested:   ", rev_renested
    assert nested == renested
    assert rev_renested == \
        (11, 10, [9, 8, 7, [6], [5], (4), ([3, 2, []]), (), ((([[]]))), []], 1)


def test_theano_function_with_nested_exprs():

    def expr_generator(a, b):
        ra = [T.pow(a[i], i) for i in range(len(a))]
        return ra, T.exp(b)

    a = [T.scalar('a%d' % i) for i in range(5)]
    b = T.scalar('b')

    f = breze.arch.util.theano_function_with_nested_exprs(
            [a, b], expr_generator(a, b))

    va = [2 for _ in a]
    vb = 3
    resa, resb = f(va, vb)

    print "va:   ", va
    print "vb:   ", vb
    print "resa: ", resa
    print "resb: ", resb

    for i in range(len(va)):
        assert np.allclose(resa[i], math.pow(va[i], i))
    assert np.allclose(resb, math.exp(vb))


