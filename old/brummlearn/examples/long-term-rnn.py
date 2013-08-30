#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This is an example which demonstrates how Krylov Subspace Descent on an
ordinary RNN can solve long range dependency sequence problems.

The problem is described in Sutskever & Marten's paper "Learning recurrent
neural networks with Hessian free optimization".
"""


import itertools
import random

import chopmunk

from breze.component import norm
from brummlearn.rnn import Rnn
from climin.initialize import sparsify_columns

import pylab
import scipy
import numpy as np
import theano
import theano.tensor as T
from zeitgeist.display import array2d_pil


# Hyper parameters.

n_inpt = 4
n_hidden = 100
n_output = 3
n_memory = 5
n_timesteps = 50
n_samples = 2**n_memory
eps = 1e-12

def softmax(inpt):
    exped = T.exp(inpt)
    return exped / (exped.sum(axis=2).dimshuffle(0, 1, 'x') + eps)

rnn = Rnn(n_inpt, n_hidden, n_output, 
          hidden_transfer='sigmoid', out_transfer=softmax, 
          loss='cross_entropy',
          optimizer='ksd')

subtarget = rnn.exprs['target'][-n_memory:]
suboutput = rnn.exprs['output'][-n_memory:]

empirical_loss = T.eq(T.gt(suboutput, 0.5), subtarget).mean()
f_empirical = rnn.function(['inpt', 'target'], empirical_loss)
f_loss = rnn.function(['inpt', 'target'], 'loss')
f_out = rnn.function(['inpt'], 'output')

# Build a dataset.

#  First, built the possible bit strings.
bitarrays = [np.array([int(j)
                       for j in ('%5i' % int(bin(i)[2:])).replace(' ', '0')])
             for i in range(n_samples)]
bitarrays = np.array(bitarrays).T


X = scipy.zeros((n_timesteps, n_samples, 4))
X[:, :, 0] = 1

X[:n_memory, :, 0] = 0
X[:n_memory, :, 1] = bitarrays
X[:n_memory, :, 2] = 1 - bitarrays

X[-n_memory, :, :] = 0, 0, 0, 1

Z = scipy.zeros((n_timesteps, n_samples, 3))
Z[:-n_memory, :, :] = 0, 0, 1
Z[-n_memory:, :, :2] = X[:n_memory, :, 1:3]

itr = rnn.iter_fit(X, Z)

for i, info in enumerate(itr):
    empirical_loss = f_empirical(X, Z)
    objective = f_loss(X, Z)
    print 'iteration #%i empirical loss %.3f cross entropy loss %.3f' % (
        i, empirical_loss, objective)

    if i > 100 or empirical_loss == 1.:
        break
