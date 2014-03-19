# -*- coding: utf-8 -*-

import numpy as np
from breze.learn import autoencoder
from breze.learn.trainer.trainer import GentleTrainer
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