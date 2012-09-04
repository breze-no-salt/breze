#! /usr/bin/env python
#! -*- coding: utf-8 -*-

__author__ = 'Justin Bayer, bayer.justin@googlemail.com'

import cPickle
import gzip
import sys

import scipy
import theano.tensor as T

from brummlearn.mlp import Mlp
from brummlearn.utils import one_hot


if __name__ == '__main__':
    n_episodes = 200
    datafile = 'mnist.pkl.gz'

    # Load data.
    try:
        with gzip.open(datafile,'rb') as f:
            train_set, valid_set, test_set = cPickle.load(f)
    except IOError:
        print 'did not find mnist data set, you can download it from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        sys.exit(1)

    X, Z = train_set
    Z = one_hot(Z, 10)
    TX, TZ = test_set
    TZ = one_hot(TZ, 10)
    VX, VZ = valid_set
    VZ = one_hot(VZ, 10)

    net = Mlp(
            784, [300], 10, 
            hidden_transfers=['tanh'],  out_transfer='softmax',
            loss='neg_cross_entropy',
            optimizer=('gd', {'steprate': .001, 'momentum': 0.9}),
            max_iter=100, verbose=True)

    targets = T.argmax(net.exprs['target'], 1)
    predictions = T.argmax(net.exprs['output_in'], 1)
    f_incorrect = net.function(['inpt', 'target'], T.neq(targets, predictions).sum())

    for info in net.iter_fit(X, Z):
        print 'incorrect samples', f_incorrect(TX, TZ)
