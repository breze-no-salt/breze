#! /usr/bin/env python
#! -*- coding: utf-8 -*-

__author__ = 'Justin Bayer, bayer.justin@googlemail.com'

import cPickle
import gzip
import sys

import scipy
from sklearn.decomposition import PCA
import pylab

from brummlearn.tsne import tsne
from brummlearn.pca import pca


if __name__ == '__main__':
    n_episodes = 1000
    ee = 100
    datafile = 'mnist.pkl.gz'
    n_components = 50
    n_samples = 500
    perplexity = 20

    # Load data.
    try:
        with gzip.open(datafile,'rb') as f:
            train_set, valid_set, test_set = cPickle.load(f)
    except IOError:
        print 'did not find mnist data set, you can download it from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        sys.exit(1)

    X, Z = train_set
    X = X[:n_samples]
    Z = Z[:n_samples]

    X -= X.mean(axis=0)
    w, b = pca(X, n_components)
    X = scipy.dot(X, w)

    # Only keep some dimensions.
    E = tsne(X, 2, perplexity=perplexity, early_exaggeration=ee,
             max_iter=n_episodes, verbose=True)

    pylab.scatter(E[:, 0], E[:, 1], c=Z, s=50)
    pylab.show()
