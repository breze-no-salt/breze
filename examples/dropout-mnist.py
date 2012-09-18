#! /usr/bin/env python
#! -*- coding: utf-8 -*-

__author__ = 'Justin Bayer, bayer.justin@googlemail.com'

import cPickle
import gzip
import sys

import numpy as np
import theano.tensor as T
import theano.tensor.nnet

from brummlearn.mlp import DropoutMlp
from brummlearn.utils import one_hot

import chopmunk

import climin.stops


def setup_logging():
    file_sink = chopmunk.jsonify(
                    chopmunk.file_sink('latest-dropout.log'))

    print_sink = chopmunk.prettyprint_sink()
    sink = chopmunk.broadcast(print_sink, file_sink)

    sink = chopmunk.dontkeep(
        sink, ['args', 'step', 'gradient', 'wrt', 'kwargs'])


    def stats(arr):
        return arr.min(), arr.max(), arr.mean(), arr.std()

    @chopmunk.coroutine
    def relevant(consumer):
        while True:
            info = (yield)
            for i in 'step', 'gradient', 'wrt':
                try:
                    cur = info[i]
                except KeyError:
                    continue
                mini, maxi, mean, std = stats(cur)

                info['%s-min' % i] = mini
                info['%s-max' % i] = maxi
                info['%s-mean' % i] = mean
                info['%s-std' % i] = std 

                del info[i]
            consumer.send(info)

    sink = relevant(sink)

    return sink


if __name__ == '__main__':
    n_epochs = 100
    batch_size = 100
    datafile = 'mnist.pkl.gz'
    p_dropout_inpt = 0.2
    p_dropout_hidden = 0.5

    # Load data.
    try:
        f = gzip.open(datafile,'rb')

    except IOError:
        print 'did not find mnist data set, you can download it from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        sys.exit(1)
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    X, Z = train_set
    Z = one_hot(Z, 10)
    TX, TZ = test_set
    TZ = one_hot(TZ, 10)
    VX, VZ = valid_set
    VZ = one_hot(VZ, 10)

    net = DropoutMlp(
            #784, [300, 300], 10,
            784, [800] * 2, 10,
            #hidden_transfers=['softsign', 'softsign'], out_transfer='softmax',
            hidden_transfers=['tanh'] * 2, out_transfer='softmax',
            loss='neg_cross_entropy',
            batch_size=batch_size,
            max_norm=15,
            p_dropout_inpt=p_dropout_hidden,
            p_dropout_hidden=p_dropout_hidden,
            max_iter=n_epochs, verbose=True)

    targets = T.argmax(net.exprs['target'], axis=1)
    predictions = T.argmax(net.exprs['output_in'], axis=1)
    incorrect = T.neq(targets, predictions).sum()
    f_incorrect_and_loss = net.function(['inpt', 'target'], ['loss', incorrect])

    f_output_in = net.function(['inpt'], 'output_in')

    every_kth = climin.stops.modulo_n_iterations(max(50000 / batch_size, 1))
    #every_kth = climin.stops.modulo_n_iterations(5)
    stop = climin.stops.after_n_iterations(n_epochs * 50000 / batch_size)

    log = setup_logging()

    print 'starting training'
    for i, info in enumerate(net.iter_fit(X, Z)):
        if every_kth(info):
            net.parameters['in_to_hidden'] *= (1 - p_dropout_inpt)
            net.parameters['hidden_to_out'] *= (1 - p_dropout_hidden)
            for j in range(len(net.n_hiddens) - 1):
                net.parameters['hidden_to_hidden_%i' % j] *= (1 - p_dropout_hidden)



            train_error, train_empirical = f_incorrect_and_loss(X, Z)
            val_error, val_empirical = f_incorrect_and_loss(VX, VZ)
            test_error, test_empirical = f_incorrect_and_loss(TX, TZ)

            info.update({
                'train_error': train_error,
                'train_empirical': train_empirical,
                'val_error': val_error,
                'val_empirical': val_empirical,
                'test_error': test_error,
                'test_empirical': test_empirical,
                'wrt': net.parameters.data})

            log.send(info)

            net.parameters['in_to_hidden'] /= (1 - p_dropout_inpt)
            net.parameters['hidden_to_out'] /= (1 - p_dropout_hidden)
            for j in range(len(net.n_hiddens) - 1):
                net.parameters['hidden_to_hidden_%i' % j] /= (1 - p_dropout_hidden)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
        if stop(info):
            print 'max iterations reached'
            break
