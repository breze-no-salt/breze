# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T

import breze.util
from breze.util import ParameterSet, Model
from tools import roughly


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

    model = Model()
    model.exprs = {'inpt': inpt, 'output': output}
    model.parameters = pars

    f = model.function(['inpt'], 'output')
    fx = model.function(['inpt'], 'output', explicit_pars=True)

    np.random.seed(1010)
    test_inpt = np.random.random((10, 2))
    correct = roughly(f(test_inpt), fx(pars.data, test_inpt))

    assert correct, 'implicit pars and explicit pars have different output'

    f1 = model.function(['inpt'], ['output'])
    f2 = model.function([inpt], ['output'])
    f3 = model.function([inpt], [output])
    f4 = model.function(['inpt'], [output])

    assert roughly(f1(test_inpt), f2(test_inpt)), "f1 and f2 don't agree"
    assert roughly(f1(test_inpt), f3(test_inpt)), "f1 and f3 don't agree"
    assert roughly(f1(test_inpt), f4(test_inpt)), "f1 and f4 don't agree"
    assert roughly(f2(test_inpt), f3(test_inpt)), "f2 and f3 don't agree"
    assert roughly(f2(test_inpt), f4(test_inpt)), "f2 and f4 don't agree"
    assert roughly(f3(test_inpt), f4(test_inpt)), "f3 and f4 don't agree"


def test_flatten():
    nested = (1, 2, [3, 4, 5, [6], [7], (8), ([9, 10, []]), (), ((([[]]))), []], 11)
    flattened = breze.util.flatten(nested)
    rev_flattened = flattened[::-1]
    renested = breze.util.unflatten(nested, flattened)
    rev_renested = breze.util.unflatten(nested, rev_flattened)
    print "nested:              ", nested
    print "flattened:           ", flattened
    print "renested:            ", renested
    print "reversed flattened:  ", rev_flattened
    print "reversed renested:   ", rev_renested
    assert nested == renested
    assert rev_renested == \
        (11, 10, [9, 8, 7, [6], [5], (4), ([3, 2, []]), (), ((([[]]))), []], 1)

