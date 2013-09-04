#!/usr/bin/env python

import numpy as np

from breze.learn.tsne import Tsne


def test_tsne():
    X = np.random.random((100, 3))
    tsne = Tsne(n_inpt=3, n_lowdim=2, perplexity=40, early_exaggeration=50,
                max_iter=10)
    E = tsne.fit_transform(X)
