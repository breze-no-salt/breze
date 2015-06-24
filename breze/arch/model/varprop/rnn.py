# -*- coding: utf-8 -*-
"""Module implementing variance propagation and fast dropout for recurrent
networks.

In this module, we will often do with multiple sequences organized into a
single Theano tensor. This tensor then has the shape of ``(t, n, d)``, where

 - ``t`` is the number of time steps,
 - ``n`` is the number of samples and
 - ``d`` is the dimensionality of each sample.

We call these "sequence tensor". Sometimes, it makes sense to flatten out the
time dimension to apply better optimized linear algebra, such as a dot product.
In that case, we will talk of a "flat sequence tensor".
"""


import theano
import theano.tensor as T
from theano.tensor.extra_ops import repeat


# TODO check documentation


def flat_time(x):
    """Return a flat sequence tensor given a sequence tensor.

    Parameters
    ----------

    x : Theano variable
        Sequence tensor: a Theano variable of shape ``(t, n ,d)``.

    Returns
    -------

    y : Theano variable
        Packed sequence tensor: a Theano variable of shape ``(t * n, d)`` with
        the same contents as ``x``.
    """
    return x.reshape((x.shape[0] * x.shape[1], x.shape[2]))


def unflat_time(x, time_steps):
    """Return a sequence tensor given a flat sequence tensor and a number of
    time steps.

    Parameters
    ----------

    x : Theano variable
        Packed sequence tensor: a Theano variable of shape ``(t * n, d)`` with
        the same contents as ``x``.

    time_steps : integer
        Number of time steps of the result.

    Returns
    -------

    y : Theano variable
        Sequence tensor: a Theano variable of shape ``(t, n ,d)``.

    """
    return x.reshape((time_steps, x.shape[0] // time_steps, x.shape[1]))


def recurrent_layer(in_mean, in_var, weights, f,
                    initial_hidden_mean, initial_hidden_var,
                    p_dropout):
    """Return a theano variable representing a recurrent layer.

    Parameters
    ----------

    in_mean : Theano variable
        Sequence tensor of shape ``(t, n ,d)``. Represents the mean of the
        input to the layer.

    in_var : Theano variable
        Sequence tensor. Represents the variance of the input to the layer.
        Either (a) same shape as the mean or (b) scalar.

    weights : Theano variable
        Theano matrix of shape ``(d, d)``. Represents the recurrent weight
        matrix the hiddens are right multiplied with.

    f : function
        Function that takes a theano variable and returns a theano variable of
        the same shape. Meant as transfer function of the layer.

    initial_hidden : Theano variable
        Theano vector of size ``d``, representing the initial hidden state.

    p_dropout : Theano variable
        Scalar representing the probability that unit is dropped out.


    Returns
    -------

    hidden_in_mean_rec : Theano variable
        Theano sequence tensor representing the mean of the hidden activations
        before the application of ``f``.

    hidden_in_var_rec : Theano variable
        Theano sequence tensor representing the varianceof the hidden
        activations before the application of ``f``.

    hidden_mean_rec : Theano variable
        Theano sequence tensor representing the mean of the hidden activations
        after the application of ``f``.

    hidden_var_rec : Theano variable
        Theano sequence tensor representing the varianceof the hidden
        activations after the application of ``f``.
    """
    def step(inpt_mean, inpt_var, him_m1, hiv_m1, hom_m1, hov_m1):
        hom = T.dot(hom_m1, weights) * p_dropout + inpt_mean

        p_keep = 1 - p_dropout
        dropout_var = p_dropout * (1 - p_dropout)

        element_var = (hov_m1 * dropout_var
                       + (hom_m1 ** 2) * dropout_var
                       + hov_m1 * p_keep ** 2)

        hov = T.dot(element_var, weights ** 2) + inpt_var

        fhom, fhov = f(hom, hov)

        return hom, hov, fhom, fhov

    if initial_hidden_mean.ndim == 1:
        initial_hidden_mean = repeat(
            initial_hidden_mean.dimshuffle('x', 0), in_mean.shape[1], axis=0)
    if initial_hidden_var.ndim == 1:
        initial_hidden_var = repeat(
            initial_hidden_var.dimshuffle('x', 0), in_mean.shape[1], axis=0)

    (hidden_in_mean_rec, hidden_in_var_rec, hidden_mean_rec, hidden_var_rec), _ = theano.scan(
        step,
        sequences=[in_mean, in_var],
        outputs_info=[T.zeros_like(in_mean[0]),
                      T.zeros_like(in_mean[0]),
                      initial_hidden_mean,
                      initial_hidden_var])

    #hidden_mean_rec, hidden_var_rec = f(
    #    hidden_in_mean_rec, hidden_in_var_rec)

    return (hidden_in_mean_rec, hidden_in_var_rec,
            hidden_mean_rec, hidden_var_rec)


def recurrent_layer_stateful(
        in_mean, in_var, weights, f, initial_hidden_mean, initial_hidden_var,
        p_dropout):
    # TODO: documentation needs to explain the stateful thing.
    """Return a theano variable representing a recurrent layer.

    Parameters
    ----------

    in_mean : Theano variable
        Sequence tensor of shape ``(t, n ,d)``. Represents the mean of the
        input to the layer.

    in_var : Theano variable
        Sequence tensor. Represents the variance of the input to the layer.
        Either (a) same shape as the mean or (b) scalar.

    weights : Theano variable
        Theano matrix of shape ``(d, d)``. Represents the recurrent weight
        matrix the hiddens are right multiplied with.

    f : function
        Function that takes a theano variable and returns a theano variable of
        the same shape. Meant as transfer function of the layer.

    initial_hidden : Theano variable
        Theano vector of size ``d``, representing the initial hidden state.

    p_dropout : Theano variable
        Scalar representing the probability that unit is dropped out.


    Returns
    -------

    hidden_in_mean_rec : Theano variable
        Theano sequence tensor representing the mean of the hidden activations
        before the application of ``f``.

    hidden_in_var_rec : Theano variable
        Theano sequence tensor representing the varianceof the hidden
        activations before the application of ``f``.

    hidden_mean_rec : Theano variable
        Theano sequence tensor representing the mean of the hidden activations
        after the application of ``f``.

    hidden_var_rec : Theano variable
        Theano sequence tensor representing the varianceof the hidden
        activations after the application of ``f``.
    """
    def step(inpt_mean, inpt_var, state_mean_tm1, state_var_tm1,
             him_m1, hiv_m1, hom_m1, hov_m1):
        hom = T.dot(hom_m1, weights) * p_dropout + inpt_mean

        p_keep = 1 - p_dropout
        dropout_var = p_dropout * (1 - p_dropout)

        element_var = (hov_m1 * dropout_var
                       + (hom_m1 ** 2) * dropout_var
                       + hov_m1 * p_keep ** 2)

        hov = T.dot(element_var, weights ** 2) + inpt_var

        state_mean_tm1 *= p_keep
        state_var_tm1 *= dropout_var ** 2

        state_mean, state_var, fhom, fhov = f(
            state_mean_tm1, state_var_tm1, hom, hov)

        return state_mean, state_var, hom, hov, fhom, fhov

    if initial_hidden_mean.ndim == 1:
        initial_hidden_mean = repeat(
            initial_hidden_mean.dimshuffle('x', 0), in_mean.shape[1], axis=0)
    if initial_hidden_var.ndim == 1:
        initial_hidden_var = repeat(
            initial_hidden_var.dimshuffle('x', 0), in_mean.shape[1], axis=0)

    (state_mean, state_var, hidden_in_mean_rec, hidden_in_var_rec,
    hidden_mean_rec, hidden_var_rec), _ = theano.scan(
        step,
        sequences=[in_mean, in_var],
        outputs_info=[T.zeros_like(initial_hidden_mean),
                      T.zeros_like(initial_hidden_mean) + 1e-8,
                      T.zeros_like(in_mean[0]),
                      T.zeros_like(in_mean[0]),
                      initial_hidden_mean,
                      initial_hidden_var])

    return (state_mean, state_var, hidden_in_mean_rec, hidden_in_var_rec,
            hidden_mean_rec, hidden_var_rec)
