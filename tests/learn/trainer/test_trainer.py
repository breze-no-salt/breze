# -*- coding: utf-8 -*-
import cPickle
import copy
import os

import numpy as np
import climin.stops

from breze.learn import mlp
from breze.learn.trainer.trainer import Trainer
from breze.learn.utils import theano_floatx
from breze.learn.trainer.score import MinibatchScore


def check_infos(info1, info2):
    for key in info1:
        if key in ('time', 'datetime'):
            continue
        if isinstance(info1[key], np.ndarray):
            assert np.allclose(info1[key], info2[key])
        elif isinstance(info1[key], list):
            for e1, e2 in zip(info1[key], info2[key]):
                if isinstance(e1, np.ndarray):
                    assert np.allclose(e1, e2)
                else:
                    assert e1 == e2
        else:
            assert info1[key] == info2[key], '%s != %s (%s)' % (
                info1[key], info2[key], key)


def test_minibatch_score_trainer():
    X = np.random.random((100, 10))
    Z = np.random.random((100, 2))
    X, Z = theano_floatx(X, Z)
    cut_size = 10

    class MyModel(mlp.Mlp):

        def score(self, X, Z):
            assert X.shape[0] <= cut_size
            return super(MyModel, self).score(X, Z)

    m = MyModel(10, [100], 2, ['tanh'], 'identity', 'squared',
                max_iter=10)

    score = MinibatchScore(cut_size, [0, 0])
    data = {
        'train': (X,Z),
        'val': (X,Z),
        'test': (X,Z)
    }
    trainer = Trainer(
        m, data=data, score=score, pause=lambda info: True, stop=lambda info: False)

    for _ in trainer.fit():
        break


def test_checkpoint_trainer():
    # Make model and data for the test.
    X = np.random.random((100, 10))
    Z = np.random.random((100, 2))
    X, Z = theano_floatx(X, Z)

    optimizer = 'rmsprop', {'step_rate': 0.0001}
    m = mlp.Mlp(10, [2], 2, ['tanh'], 'identity', 'squared', max_iter=10,
                optimizer=optimizer)

    # Train the mdoel with a trainer for 2 epochs.
    data = {
        'train': (X,Z),
        'val': (X,Z),
        'test': (X,Z)
    }
    t = Trainer(
        m,
        data=data,
        stop=climin.stops.AfterNIterations(2),
        pause=climin.stops.always)
    t.fit()

    # Make a copy of the trainer.
    t2 = copy.deepcopy(t)
    print type(t.current_info)
    print type(t2.current_info)
    intermediate_pars = t2.model.parameters.data.copy()
    intermediate_info = t2.current_info.copy()

    # Train original for 2 more epochs.
    t.stop = climin.stops.AfterNIterations(4)
    t.fit()

    # Check that the snapshot has not changed
    assert np.all(t2.model.parameters.data == intermediate_pars)

    final_pars = t.model.parameters.data.copy()
    final_info = t.current_info.copy()

    check_infos(intermediate_info, t2.current_info)

    t2.stop = climin.stops.AfterNIterations(4)
    t2.fit()
    check_infos(final_info, t2.current_info)

    assert np.allclose(final_pars, t2.model.parameters.data)

    t_pickled = cPickle.dumps(t2)
    t_unpickled = cPickle.loads(t_pickled)
    t.stop = climin.stops.AfterNIterations(4)

    t_unpickled.fit()

    assert np.allclose(
        final_pars, t_unpickled.model.parameters.data, atol=5.e-3)


def test_training_continuation():
    # Make model and data for the test.
    X = np.random.random((100, 10))
    Z = np.random.random((100, 2))
    X, Z = theano_floatx(X, Z)

    optimizer = 'rmsprop', {'step_rate': 0.0001}
    m = mlp.Mlp(10, [2], 2, ['tanh'], 'identity', 'squared', max_iter=10,
                optimizer=optimizer)

    # Train the mdoel with a trainer for 2 epochs.
    stopper = climin.stops.OnSignal()
    stops = climin.stops.Any([stopper, climin.stops.AfterNIterations(5)])
    data = {
        'train': (X,Z),
        'val': (X,Z),
        'test': (X,Z)
    }
    t = Trainer(
        m,
        data=data,
        stop=stops,
        pause=climin.stops.always)

    for info in t.fit():
        os.kill(os.getpid(), stopper.sig)

    assert info['n_iter'] == 1
