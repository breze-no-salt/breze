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

from breze.model.neural import MultilayerPerceptron as MLP

from utils import tile_raster_images, one_hot


# Hyperparameters.
batch_size = 1000

# Make data.
f = gzip.open('mnist.pkl.gz', 'rb')
(X, Z), (VX, VZ), (TX, TZ) = cPickle.load(f)
f.close()
Z = one_hot(Z, 10)
VZ = one_hot(VZ, 10)
TZ = one_hot(TZ, 10)

slices = draw_mini_slices(X.shape[0], batch_size)
args = (([X[s], Z[s]], {}) for s in slices)

# Build NN
nn = MLP(784, 300, 10, 'tanh', 'softmax', 'cross_entropy') 
f = nn.function(['inpt', 'target'], 'loss', explicit_pars=True)
d_loss_wrt_pars = T.grad(nn.exprs['loss'], nn.parameters.flat)
fprime = nn.function(['inpt', 'target'], d_loss_wrt_pars, explicit_pars=True)

nn.parameters.data[:] = 0.01*np.random.randn(*nn.parameters.data.shape)

pred = nn.function(['inpt'], 'output', explicit_pars=True)

def logfunc(info): print info

def zero_one(prediction, target):
    return 1 - np.sum(np.argmax(prediction, axis=1) == np.argmax(target, axis=1))/(1. *target.shape[0])

# Optimize!
opt = Lbfgs(nn.parameters.data, f, fprime, args=args)
for i, info in enumerate(opt):
    print i
    if (i+1) % 50 == 0:
        #loss = f(logreg.parameters.data, X, Z)
        val_loss = f(nn.parameters.data, VX, VZ)
        test_loss = f(nn.parameters.data, TX, TZ)
        print 'validate loss', val_loss, 'test loss', test_loss
        print 'zero_one', zero_one(pred(nn.parameters.data, TX), TZ)

    if i > 1000:
        break
