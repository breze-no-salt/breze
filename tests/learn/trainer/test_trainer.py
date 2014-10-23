# -*- coding: utf-8 -*-
import cPickle
import copy
import os

import numpy as np
import climin.stops

from breze.learn import autoencoder
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
            assert info1[key] == info2[key], '%s != %s (%s)' % (info1[key], info2[key], key)


def test_minibatch_score_trainer():
    X = np.random.random((100, 10))
    X, = theano_floatx(X)
    cut_size = 10

    class MyAutoEncoder(autoencoder.AutoEncoder):

        def score(self, X):
            assert X.shape[0] <= cut_size
            return super(MyAutoEncoder, self).score(X)

    m = MyAutoEncoder(10, [100], ['tanh'], 'identity', 'squared',
                      tied_weights=True, max_iter=10)

    score = MinibatchScore(cut_size, [0])
    trainer = Trainer(
        m, score=score, pause=lambda info: True, stop=lambda info: False)
    trainer.eval_data = {'val': (X,)}
    trainer.val_key = 'val'

    for _ in trainer.iter_fit(X):
        break


def test_checkpoint_trainer():
    # Make model and data for the test.
    X = np.random.random((10, 2))
    X, = theano_floatx(X)
    optimizer = 'rmsprop', {'step_rate': 0.0001}
    m = autoencoder.AutoEncoder(2, [2], ['tanh'], 'identity', 'squared',
                                tied_weights=True, max_iter=10,
                                optimizer=optimizer)

    # Train the mdoel with a trainer for 2 epochs.
    t = Trainer(
        m,
        stop=climin.stops.AfterNIterations(2),
        pause=climin.stops.always)
    t.val_key = 'val'
    t.eval_data = {'val': (X,)}
    t.fit(X)

    # Make a copy of the trainer.
    t2 = copy.deepcopy(t)
    intermediate_pars = t2.model.parameters.data.copy()
    intermediate_info = t2.current_info.copy()

    # Train original for 2 more epochs.
    t.stop = climin.stops.AfterNIterations(4)
    t.fit(X)

    # Check that the snapshot has not changed
    assert np.all(t2.model.parameters.data == intermediate_pars)

    final_pars = t.model.parameters.data.copy()
    final_info = t.current_info.copy()

    check_infos(intermediate_info, t2.current_info)

    t2.stop = climin.stops.AfterNIterations(4)
    t2.fit(X)
    check_infos(final_info, t2.current_info)

    assert np.allclose(final_pars, t2.model.parameters.data)

    t_pickled = cPickle.dumps(t2)
    t_unpickled = cPickle.loads(t_pickled)
    t.stop = climin.stops.AfterNIterations(4)

    t_unpickled.fit(X)

    assert np.allclose(final_pars, t_unpickled.model.parameters.data, atol=5.e-3)


def test_training_continuation():
    # Make model and data for the test.
    X = np.random.random((10, 2))
    X, = theano_floatx(X)
    optimizer = 'gd'
    m = autoencoder.AutoEncoder(2, [2], ['tanh'], 'identity', 'squared',
                                tied_weights=True, max_iter=10,
                                optimizer=optimizer)

    # Train the mdoel with a trainer for 2 epochs.
    stopper = climin.stops.OnSignal()
    print stopper.sig
    stops = climin.stops.Any([stopper, climin.stops.AfterNIterations(5)])
    t = Trainer(
        m,
        stop=stops,
        pause=climin.stops.always)

    t.val_key = 'val'
    t.eval_data = {'val': (X,)}
    killed = False
    for info in t.iter_fit(X):
        os.kill(os.getpid(), stopper.sig)

    assert info['n_iter'] == 1
