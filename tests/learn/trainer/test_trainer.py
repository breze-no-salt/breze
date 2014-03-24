# -*- coding: utf-8 -*-
import cPickle

import numpy as np
from breze.learn import autoencoder
from breze.learn.trainer.trainer import Trainer, SnapshotTrainer
from breze.learn.utils import theano_floatx

from breze.learn.trainer.score import MinibatchScore


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
    trainer = Trainer('spam', m, score=score)

    for _ in trainer.fit((X,), {'val': (X,)}, lambda info: False, lambda info: True):
        break


def test_checkpoint_trainer():

    def check_infos(info1, info2):
        for key in info1:
            if key == 'time':
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
                assert info1[key] == info2[key]

    X = np.random.random((100, 10))
    X, = theano_floatx(X)
    optimizer = 'rmsprop', {'step_rate': 0.0001, 'momentum': 0.9,
                            'decay': 0.9}
    m = autoencoder.AutoEncoder(10, [100], ['tanh'], 'identity', 'squared',
                                tied_weights=True, max_iter=10,
                                optimizer=optimizer)
    t = SnapshotTrainer('eggs', m)
    for _ in t.fit((X,), {'val': (X,)}, lambda info: info['n_iter'] >= 1000,
                   lambda info: True):
        pass
    snapshot = t.provide_snapshot(copy=True)
    intermediate_pars = t.model.parameters.data.copy()
    intermediate_info = t.current_info.copy()
    for _ in t.fit((X,), {'val': (X,)}, lambda info: info['n_iter'] >= 1000,
                   lambda info: True):
        pass
    #Check that the snapshot has not changed
    assert np.all(snapshot['model'].parameters.data == intermediate_pars)
    final_pars = t.model.parameters.data.copy()
    final_info = t.current_info.copy()
    t = SnapshotTrainer.load_trainer(snapshot)
    check_infos(intermediate_info, t.current_info)
    assert np.all(intermediate_pars == t.model.parameters.data)

    for _ in t.fit((X,), {'val': (X,)}, lambda info: info['n_iter'] >= 1000,
                   lambda info: True):
        pass
    check_infos(final_info, t.current_info)

    assert np.allclose(final_pars, t.model.parameters.data)

    s = cPickle.dumps(snapshot)
    snapshot_from_pickle = cPickle.loads(s)

    t = SnapshotTrainer.load_trainer(snapshot_from_pickle)
    for _ in t.fit((X,), {'val': (X,)}, lambda info: info['n_iter'] >= 1000,
                   lambda info: True):
        pass

    assert np.allclose(final_pars, t.model.parameters.data, atol=5.e-3)
