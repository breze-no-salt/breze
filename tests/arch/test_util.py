# -*- coding: utf-8 -*-


import cPickle

import numpy as np
import theano
import theano.tensor as T
import math

import breze.arch.util
from breze.arch.util import (ParameterSet, Model, array_partition_views,
                             n_pars_by_partition)


def test_parameter_set_init_declare():
    pars = ParameterSet()

    matrix = pars.declare((10, 10))
    vector = pars.declare((10,))
    pars.alloc()

    assert pars.data.shape == (110,), 'wrong size for flat pars allocated'
    assert (pars[matrix].shape == (10, 10)), ('wrong size for 2d array in pars '
                                                'allocated')
    assert (pars[vector].shape == (10,)), ('wrong size for 1d array in pars '
                                             'allocated')


def test_parameter_set_init_overwrite():
    pars = ParameterSet()

    matrix = pars.declare((10, 10))
    pars.alloc()

    pars[matrix] = np.eye(10)
    assert np.allclose(pars.data.reshape(pars[matrix].shape), pars[matrix])



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


def test_model_function_mode():
    pars = ParameterSet(weights=(2, 3))
    inpt = T.matrix()
    output = T.dot(inpt, pars.weights)
    pars.data[...] = np.random.standard_normal(pars.data.shape)

    model = Model()
    model.exprs = {'inpt': inpt, 'output': output}
    model.parameters = pars

    mode = theano.Mode()

    f = model.function(['inpt'], 'output', mode=mode)
    actual_mode = f.theano_func.maker.mode
    assert actual_mode is mode, 'wrong mode: %s' % actual_mode

    model.mode = theano.Mode()
    f = model.function(['inpt'], 'output')
    actual_mode = f.theano_func.maker.mode

    # Maybe a weird way to compare modes, but it seems to get the job done.
    equal = actual_mode.__dict__ == mode.__dict__
    assert equal, 'wrong mode: (%s != %s)' % (actual_mode, mode)


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


def test_pickling_models():
    ma = T.matrix()
    m = Model()
    m.parameters = ParameterSet(bla=2)
    m.exprs = {'m_sqrd': m.parameters.bla.sum() * ma.sum()}
    m.f = m.function([ma], 'm_sqrd', explicit_pars=False)

    cPickle.dumps(m)


def test_nested_pars():
    spec = {
        'a': [2, 3],
        'b': {
            'a': (10, 10),
            'b': (2,)
        }
    }

    ps = ParameterSet(**spec)
    assert ps.data.size == 2 + 3 + 10 * 10 + 2


def test_array_partition_views():
    flat = np.arange(14).astype('float64')
    partition = make_dictlist()

    views = array_partition_views(flat, partition)

    assert np.allclose(views['bar'], np.arange(4).reshape((2, 2)))
    assert np.allclose(views['fank']['fenk'][0], 4.)
    assert np.allclose(views['fank']['funk'], np.array([6, 7]).reshape((2, 1)))


def make_dictlist():
    return {
        'bar': (2, 2),
        'fank': {
            'fenk': [1, 1],
            'funk': (2, 1),
        },
        'foo': 4,
        'fink': [1, 1],
    }


def test_n_pars_by_partition():
    tree = make_dictlist()
    assert n_pars_by_partition(tree) == 14


def test_nested_parameter_set():
    spec = make_dictlist()
    p = ParameterSet(**spec)

    assert p['bar'].shape == (2, 2)
    assert p.bar.ndim == 2

    assert p.fank.fenk[0].ndim == 1
    assert p['fank']['funk'].shape == (2, 1)


def test_nested_exprs():
    ma = T.matrix()
    m = Model()
    m.parameters = ParameterSet(bla=2)
    m.parameters['bla'][...] = 1, 2
    m.exprs = {
        'norms': {
            'l1': abs(ma).sum(),
            'l2': T.sqrt((ma ** 2).sum()),
        },
        'ma_multiplied': [ma, 2 * ma],
        'bla': m.parameters.bla,
        'blubb': 1,
    }

    f = m.function([], 'bla', explicit_pars=False, on_unused_input='ignore')
    assert np.allclose(f(), [1, 2])

    f = m.function([ma], ('norms', 'l1'),
                   explicit_pars=False,
                   on_unused_input='ignore')

    assert f([[-1, 1]]) == 2

    f = m.function([ma], ('norms', 'l2'),
                   explicit_pars=False,
                   on_unused_input='ignore')

    assert np.allclose(f([[-1, 1]]), np.sqrt(2.))

    f = m.function([ma], ('ma_multiplied', 0),
                   explicit_pars=False,
                   on_unused_input='ignore')
    assert np.allclose(f([[-1, 1]]), [-1, 1])

    f = m.function([ma], ('ma_multiplied', 1),
                   explicit_pars=False,
                   on_unused_input='ignore')
    assert np.allclose(f([[-1, 1]]), [-2, 2])
