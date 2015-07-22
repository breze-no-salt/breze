# -*- coding: utf-8 -*-


import theano.tensor as T

from breze.arch.construct.neural import distributions


def test_mlp_diag_gauss():
    inpt = T.tensor3('inpt')
    distributions.MlpDiagGauss(
        inpt,
        3, [4], 5,
        ['tanh'])


def test_mlp_bernoulli():
    inpt = T.tensor3('inpt')
    distributions.MlpBernoulli(
        inpt,
        3, [4], 5,
        ['tanh'])


def test_fd_mlp_diag_gauss():
    inpt = T.tensor3('inpt')
    distributions.FastDropoutMlpDiagGauss(
        inpt,
        3, [4], 5,
        ['tanh'],
        p_dropout_inpt=.1,
        p_dropout_hiddens=[.1]
    )


def test_fd_mlp_bernoulli():
    inpt = T.tensor3('inpt')
    distributions.FastDropoutMlpBernoulli(
        inpt,
        3, [4], 5,
        ['tanh'],
        p_dropout_inpt=.1,
        p_dropout_hiddens=[.1]
    )


def test_rnn_diag_gauss():
    inpt = T.tensor3('inpt')
    distributions.RnnDiagGauss(
        inpt,
        3, [4], 5,
        ['tanh'])


def test_rnn_bernoulli():
    inpt = T.tensor3('inpt')
    distributions.RnnBernoulli(
        inpt,
        3, [4], 5,
        ['tanh'])


def test_fdrnn_diag_gauss():
    inpt = T.tensor3('inpt')
    distributions.FastDropoutRnnDiagGauss(
        inpt,
        3, [4], 5,
        ['tanh'],
        p_dropout_inpt=.1,
        p_dropout_hiddens=[.1],
    )


def test_fdrnn_bernoulli():
    inpt = T.tensor3('inpt')
    distributions.FastDropoutRnnBernoulli(
        inpt,
        3, [4], 5,
        ['tanh'],
        p_dropout_inpt=.1,
        p_dropout_hiddens=[.1],
    )
