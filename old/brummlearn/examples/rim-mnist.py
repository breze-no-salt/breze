#! /usr/bin/env python
#! -*- coding: utf-8 -*-

__author__ = 'Justin Bayer, bayer.justin@googlemail.com'

import cPickle
import gzip
import sys

import numpy as np
import theano.tensor as T
import Image as pil

from brummlearn.rim import Rim
from brummlearn.utils import tile_raster_images


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

    X, _ = train_set
    TX, _ = test_set
    VX, _ = valid_set

    rim = Rim(
            784, 25, 0.05,
            max_iter=100, verbose=True)

    np.set_printoptions(precision=3)
    for i, info in enumerate(rim.iter_fit(X)):
        print rim.transform(TX).mean(axis=0)

        img_arr = tile_raster_images(
            rim.parameters['in_to_out'].T, (28, 28), (5, 5)
            ).astype('uint8')

        img = pil.fromarray(img_arr)
        img.save('%i-rim-clusters.png' % i)

        if i > rim.max_iter:
            break



