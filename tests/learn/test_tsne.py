#!/usr/bin/env python

import numpy as np
import theano

from breze.learn.tsne import Tsne
from breze.learn.utils import theano_floatx



def test_tsne():
    theano.config.compute_test_value = 'raise'
    X = np.random.random((10, 3)).astype(theano.config.floatX)
    X, = theano_floatx(X)

    tsne = Tsne(n_inpt=3, n_lowdim=2, perplexity=40, early_exaggeration=50,
                max_iter=10)
    E = tsne.fit_transform(X)
