# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T

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
