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


if __name__ == '__main__':
    n_episodes = 200
    ee = 100
    datafile = 'mnist.pkl.gz'
    n_components = 50
    n_samples = 1000
    perplexity = 20

    # Load data.
    with gzip.open(datafile,'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
        X, Z = train_set
        X = X[:n_samples]
        Z = Z[:n_samples]

    pca = PCA(n_components, whiten=True)
    X = pca.fit_transform(X)

    # Only keep some dimensions.
    E = tsne(X, 2, perplexity=perplexity, early_exaggeration=ee,
             max_iter=n_episodes)

    pylab.scatter(E[:, 0], E[:, 1], c=Z, s=50)
    pylab.show()
