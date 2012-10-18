# -*- coding: utf-8 -*-


import math
import json
import random

import numpy as np

from brummlearn.mlp import Mlp
from brummlearn import hpsearch
from brummlearn.data import one_hot

import climin.stops


def iris_data(fn):
    with open(fn) as fp:
        lines = fp.readlines()
    # Remove whitespace.
    lines = [i.strip() for i in lines]
    # Remove empty lines.
    lines = [i for i in lines if i]
    # Split by comma.
    lines = [i.split(',') for i in lines]
    # Inputs are the first four elements.
    inpts = [i[:4] for i in lines]
    # Labels are the last.
    labels = [i[-1] for i in lines]

    # Make arrays out of the inputs, one row per sample.
    X = np.empty((150, 4))
    X[:] = inpts

    # Make integers array out of label strings.
    #
    # We do this by first creating a set out of all labels to remove
    # any duplicates. Then we create a dictionary which maps label
    # names to an index. Afterwards, we loop over all labels and
    # assign the corresponding integer to that field in the label array z.
    z = np.empty(150)
    label_names = sorted(set(labels))
    label_to_idx = dict((j, i) for i, j in enumerate(label_names))

    for i, label in enumerate(labels):
        z[i] = label_to_idx[label]

    return X, z

X, Z = iris_data('iris.data')
Z = one_hot(Z.astype('int'), 3)

idxs = set(range(X.shape[0]))
train_idxs = random.sample(idxs, X.shape[0] / 2)
test_idxs = list(idxs - set(train_idxs))

TX, TZ = X[test_idxs], Z[test_idxs]
X, Z = X[train_idxs], Z[train_idxs]


def f(step_rate, momentum=0, n_hidden=10, batch_size=10, par_std=0.1):
    opt = 'gd', {'steprate': step_rate, 'momentum': momentum}
    net = Mlp(4, [n_hidden], 3, hidden_transfers=['sigmoid'],
              out_transfer='softmax', loss='neg_cross_entropy',
              optimizer=opt, batch_size=batch_size)
    net.parameters.data[:] = np.random.normal(
        0, par_std, net.parameters.data.shape)

    max_iter = 1000

    stop = climin.stops.any_([
        climin.stops.time_elapsed(10),
        climin.stops.after_n_iterations(
            int(math.floor(max_iter * X.shape[0] * 1. / batch_size)))
    ])

    f_loss = net.function(['inpt', 'target'], 'loss')
    for i in net.iter_fit(X, Z):
        if stop(i):
            break

    loss = f_loss(TX, TZ)

    # Check for NaN.
    if np.isnan(loss):
        loss = 1E6

    return loss

def main():
    s = hpsearch.SearchSpace()
    s.add('step_rate', hpsearch.Uniform(0, .5))
    s.add('par_std', hpsearch.Uniform(0, 1))
    s.add('momentum', hpsearch.Uniform(0, .99))
    s.add('n_hidden', hpsearch.Uniform(10, 25, intify=True))
    s.add('batch_size', hpsearch.Uniform(1, 150, intify=True))

    strategy = 'gp'

    if strategy == 'random':
        searcher = hpsearch.RandomSearcher(s.seed_size)
    elif strategy == 'gp':
        searcher = hpsearch.GaussianProcessSearcher(s.seed_size, 50)
    elif strategy == 'rf':
        searcher = hpsearch.RandomForestSearcher(s.seed_size, 50)
    else:
        print 'unknown strategy'
        sys.exit(1)
    losses = []
    for i in range(200):
        handle, candidate = searcher.pull_candidate()
        kwargs = s.transform(candidate)
        print '-' * 20
        print 'Iteration:', i
        print 'Evaluating:', kwargs

        loss = f(**kwargs)
        searcher.push_result(handle, loss)
        losses.append(loss)

        print 'Last: %.4f Best: %5.4f' % (loss, min(losses))

    for h, c, l in searcher.results:
        print s.transform(c), l

    infos = []
    for handle, pars, loss in searcher.results:
        info = searcher.transform(pars)
        info['loss'] = loss
    with open('latest-%s.json' % strategy, 'w') as fp:
        json.dump(infos, fp)



if __name__ == '__main__':
    main()


