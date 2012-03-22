# -*- coding: utf-8 -*-

import cPickle
import itertools
import gzip

import Image as pil
import numpy as np
import theano
import theano.tensor as T

from climin import Lbfgs
from climin.util import draw_mini_slices

from breze.model.feature.sparsefiltering import SparseFiltering 

from utils import tile_raster_images


# Hyperparameters.
batch_size = 50000

# Make data.
f = gzip.open('mnist.pkl.gz', 'rb')
(X, Z), (VX, VZ), (TX, TZ) = cPickle.load(f)
f.close()

slices = draw_mini_slices(X.shape[0], batch_size)
args = (([X[s]], {}) for s in slices)

# Make feature extractor.
sf = SparseFiltering(784, 64, abs(x))
f = sf.function(['inpt'], 'loss', explicit_pars=True)
d_loss_wrt_pars = T.grad(sf.exprs['loss'], sf.parameters.flat)
fprime = sf.function(['inpt'], d_loss_wrt_pars, explicit_pars=True)

sf.parameters.data[:] = np.random.standard_normal(sf.parameters.data.shape)


def logfunc(info): print info

# Optimize!
opt = Lbfgs(sf.parameters.data, f, fprime, args=args)
for i, info in enumerate(opt):
    loss = f(sf.parameters.data, X)
    val_loss = f(sf.parameters.data, VX)
    test_loss = f(sf.parameters.data, TX)
    print 'loss', loss, 'validate loss', val_loss, 'test loss', test_loss

    # Visualize filters.
    W = sf.parameters['inpt_to_feature']
    A = tile_raster_images(W.T, (28, 28), (8, 8)).astype('float64')
    pilimage = pil.fromarray(A).convert('RGB')
    pilimage.save('sf-mnist-filters-%i.png' % i)

    if i > 1000:
        break
