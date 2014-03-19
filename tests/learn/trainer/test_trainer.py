# -*- coding: utf-8 -*-
import cPickle

import numpy as np
from breze.learn import autoencoder
from breze.learn.trainer.trainer import GentleTrainer, CheckpointTrainer
from breze.learn.utils import theano_floatx


def test_gentle_trainer():
    X = np.random.random((100, 10))
    X, = theano_floatx(X)
    cut_size = 10
    class MyAutoEncoder(autoencoder.AutoEncoder):

        def score(self, X):
            print 'calling score with shape', X.shape
            assert X.shape[0] <= cut_size
            return super(MyAutoEncoder, self).score(X)

    m = MyAutoEncoder(10, [100], ['tanh'], 'identity', 'squared',
                      tied_weights=True, max_iter=10)
    gt = GentleTrainer('spam', m, cut_size, [0])
    for _ in gt.fit((X,), {'val': (X,)}, lambda info: False, lambda info: True):
        break

def test_checkpoint_trainer():
    X = np.random.random((100, 10))
    X, = theano_floatx(X)

    m = autoencoder.AutoEncoder(10, [100], ['tanh'], 'identity', 'squared',
                                tied_weights=True, max_iter=10)
    t = CheckpointTrainer('eggs', m)
    for _ in t.fit((X,), {'val': (X,)}, lambda info: info['n_iter'] >= 10, lambda info: True):
        pass
    c = t.save_state()
    for _ in t.fit((X,), {'val': (X,)}, lambda info: info['n_iter'] >= 10, lambda info: True):
        pass
    final_pars = m.parameters.data.copy()
    t.load_state(c)
    for _ in t.fit((X,), {'val': (X,)}, lambda info: info['n_iter'] >= 10, lambda info: True):
        pass
    assert np.allclose(final_pars, m.parameters.data)

    s = cPickle.dumps(c)
    c1 = cPickle.loads(s)

    t.load_state(c1)
    for _ in t.fit((X,), {'val': (X,)}, lambda info: info['n_iter'] >= 10, lambda info: True):
        pass
    assert np.allclose(final_pars, m.parameters.data)
