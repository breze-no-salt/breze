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

from ...util import lookup
from ...component.varprop import transfer, loss as loss_
import mlp


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


def forward_layer(in_mean, in_var, weights, mean_bias, var_bias_sqrt,
                  f, p_dropout):
    """Return a theano variable representing a simple (i.e. non-recurrent
    forward layer.

    Parameters
    ----------

    in_mean : Theano variable
        Sequence tensor of shape ``(t, n ,d)``. Represents the mean of the
        input to the layer.

    in_var : Theano variable
        Sequence tensor. Represents the variance of the input to the layer.
        Either (a) same shape as the mean or (b) scalar.

    weights : Theano variable
        Theano matrix of shape ``(d, h)``. Represents the weights by which the
        input is right multiplied with.

    mean_bias : Theano variable
        Theano vector of size ``h``. Bias for the activation of the output of
        the layer.

    var_bias_sqrt : Theano variable
        Theano vector of size ``h`` or scalar. Bias for the variance of the
        output, which is multiplied by the square of this number.

    f : function
        Function that takes a theano variable and returns a theano variable of
        the same shape. Meant as transfer function of the layer.

    p_dropout : Theano variable
        Scalar representing the probability that unit is dropped out.


    Returns
    -------

    omi : Theano variable
        Mean of the output before the activation of ``f``.

    ovi : Theano variable
        Variance of the output before the activation of ``f``.

    omo : Theano variable
        Mean of the output after the activation of ``f``.

    ovo : Theano variable
        Variance of the output after the activation of ``f``.
    """
    in_mean_flat = flat_time(in_mean)
    in_var_flat = flat_time(in_var)

    omi_flat, ovi_flat, omo_flat, ovo_flat = mlp.mean_var_forward(
        in_mean_flat, in_var_flat, weights, mean_bias, var_bias_sqrt,
        f, p_dropout)

    omi = unflat_time(omi_flat, in_mean.shape[0])
    ovi = unflat_time(ovi_flat, in_mean.shape[0])
    omo = unflat_time(omo_flat, in_mean.shape[0])
    ovo = unflat_time(ovo_flat, in_mean.shape[0])

    return omi, ovi, omo, ovo


def int_forward_layer(in_mean, in_var, weights, mean_bias, var_bias_sqrt,
                      f, p_dropout):
    """Return a theano variable representing a simple (i.e. non-recurrent
    forward layer where the input is symbolic, e.g. has one of K possible
    values.
    Parameters
    ----------

    in_mean : Theano variable
        Sequence tensor of shape ``(t, n)`` and type int. Represents which of
        the ``d`` input dimensionalities is set to 1, all other are zero.

    in_var : Theano variable
        Sequence tensor. Represents the variance of the input to the layer.
        Either (a) same shape as the mean or (b) scalar.

    weights : Theano variable
        Theano matrix of shape ``(d, h)``. Represents the weights by which the
        input is right multiplied with.

    mean_bias : Theano variable
        Theano vector of size ``h``. Bias for the activation of the output of
        the layer.

    var_bias_sqrt : Theano variable
        Theano vector of size ``h`` or scalar. Bias for the variance of the
        output, which is multiplied by the square of this number.

    f : function
        Function that takes a theano variable and returns a theano variable of
        the same shape. Meant as transfer function of the layer.

    p_dropout : Theano variable
        Scalar representing the probability that unit is dropped out.


    Returns
    -------

    omi : Theano variable
        Mean of the output before the activation of ``f``.

    ovi : Theano variable
        Variance of the output before the activation of ``f``.

    omo : Theano variable
        Mean of the output after the activation of ``f``.

    ovo : Theano variable
        Variance of the output after the activation of ``f``.
    """
    in_mean_flat = in_mean.flatten()
    in_var_flat = in_var.flatten()

    omi_flat, ovi_flat, omo_flat, ovo_flat = mlp.int_mean_var_forward(
        in_mean_flat, in_var_flat, weights, mean_bias, var_bias_sqrt,
        f, p_dropout)

    omi = omi_flat.reshape((in_mean.shape[0], in_mean.shape[1], weights.shape[1]))
    ovi = ovi_flat.reshape((in_mean.shape[0], in_mean.shape[1], weights.shape[1]))
    omo = omo_flat.reshape((in_mean.shape[0], in_mean.shape[1], weights.shape[1]))
    ovo = ovo_flat.reshape((in_mean.shape[0], in_mean.shape[1], weights.shape[1]))

    omi = T.cast(omi, 'float32')
    ovi = T.cast(ovi, 'float32')
    omo = T.cast(omo, 'float32')
    ovo = T.cast(ovo, 'float32')

    return omi, ovi, omo, ovo


def recurrent_layer(in_mean, in_var, weights, f, initial_hidden,
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

    initial_hidden_mean = repeat(initial_hidden.dimshuffle('x', 0), in_mean.shape[1], axis=0)

    initial_hidden_var = T.zeros_like(initial_hidden_mean) + 1e-8

    (hidden_in_mean_rec, hidden_in_var_rec, hidden_mean_rec, hidden_var_rec), _ = theano.scan(
        step,
        sequences=[in_mean, in_var],
        outputs_info=[T.zeros_like(initial_hidden_mean),
                      T.zeros_like(initial_hidden_var),
                      initial_hidden_mean,
                      initial_hidden_var])

    #hidden_mean_rec, hidden_var_rec = f(
    #    hidden_in_mean_rec, hidden_in_var_rec)

    return (hidden_in_mean_rec, hidden_in_var_rec,
            hidden_mean_rec, hidden_var_rec)


def exprs(inpt_mean, inpt_var, in_to_hidden, hidden_to_hiddens, hidden_to_out,
          hidden_biases, hidden_var_scales_sqrt, initial_hiddens, recurrents,
          out_bias, out_var_scale_sqrt, hidden_transfers, out_transfer,
          in_to_out=None, skip_to_outs=None, p_dropouts=None,
          hotk_inpt=False):
    """Return a dictionary containing Theano expressions for various components
    of a recurrent network with variance propagation.

    Parameters
    ----------

    inpt_mean : Theano variable
        Represents the mean of the input sequences as a sequence tensor.

    inpt_var : Theano variable
        Representes the variance of the input sequences as a sequence tensor
        Can a be a scalar as well. (E.g. 1e-8 if no variance is desired at
        this point.)

    in_to_hidden : Theano variable
        Matrix representing the map from input to the first hidden layer.

    hidden_to_hiddens : list of Theano variables
        List of matrices representing the maps between the hiddens.

    hidden_to_out : Theano variable
        Matrix representing the map from the last hidden layer to the output
        layer.

    hidden_biases : list of Theano variables
        Biases for the hidden layers.

    hidden_var_scales_sqrt : Theano variable
        Biases for the variances. See ``forward_layer`` for an exact description
        of what it does.

    initial_hiddens : list of Theano variables
        List of vectors representing the initial hidden states.

    recurrents : list of Theano variables
        List of matrices representing the recurrent weight matrices.

    out_bias : Theano variable
        Bias vector of the output layer.

    hidden_transfers : list of funtions or strings
        List of transfer functions for the layers. Each element is either a
        function that given a mean and a variance sequence tensor produces
        equally shaped mean and variance tensors or a string pointing to a
        function in ``breze.arch.component.varprop.transfer``.

    out_transfer : Theano variable
        Function or string of the form described for ``hidden_transfers``.

    p_dropouts : List of scalars
        Each element in this list represents the probability to drop out an
        individual unit in the corresponding layer.
        The list should contain N+1 items, where N is the number of hidden
        layers. If N+2 items are contained, the last element is used to
        drop out units from hidden to out, while the one before is used to drop
        out units from hidden to hidden.

    Returns
    -------

    exprs : dictionary
       Map of strings to Theano expressions.

       Keys are:

         - ``hidden_in_mean_*``: pre-synaptic mean of layer,
         - ``hidden_in_var_0``: pre-synaptic variance of layer,
         - ``hidden_mean_0``: post-synaptic mean of layer,
         - ``hidden_var_0``: post-synaptic variance of layer,
         - ``inpt_mean``: mean of the input,
         - ``inpt_var``: variance of the input
         - ``output_in_mean``: pre-synaptic mean of output,
         - ``output_in_var``: pre-synptic variance of output,
         - ``output_mean``: post-synaptic mean of output,
         - ``output_var``: post-synaptic variance of output,
         - ``output``: concatenation of mean and variance of output
    """
    # TODO add skip to outs docs
    # TODO: add pooling
    # TODO: add leaky integration
    exprs = {}

    f_hiddens = [lookup(i, transfer) for i in hidden_transfers]
    f_output = lookup(out_transfer, transfer)

    if inpt_var.ndim != 3:
        # Scalar
        inpt_var = T.ones_like(inpt_mean) * inpt_var

    if hotk_inpt:
        hmi, hvi, hmo, hvo = int_forward_layer(
            inpt_mean, inpt_var, in_to_hidden,
            hidden_biases[0], hidden_var_scales_sqrt[0],
            f_hiddens[0], p_dropouts[0])
    else:
        hmi, hvi, hmo, hvo = forward_layer(
            inpt_mean, inpt_var, in_to_hidden,
            hidden_biases[0], hidden_var_scales_sqrt[0],
            f_hiddens[0], p_dropouts[0])

    hmi_rec, hvi_rec, hmo_rec, hvo_rec = recurrent_layer(
        hmi, hvi, recurrents[0], f_hiddens[0], initial_hiddens[0],
        p_dropouts[1])

    exprs.update({
        'hidden_in_mean_0': hmi_rec,
        'hidden_in_var_0': hvi_rec,
        'hidden_mean_0': hmo_rec,
        'hidden_var_0': hvo_rec
    })

    zipped = zip(
        hidden_to_hiddens, hidden_biases[1:], hidden_var_scales_sqrt[1:],
        recurrents[1:], f_hiddens[1:], initial_hiddens[1:], p_dropouts[1:])

    for i, (w, b, vb, r, t, j, d) in enumerate(zipped):
        hmo_rec_m1, hvo_rec_m1 = hmo_rec, hvo_rec

        hmi, hvi, hmo, hvo = forward_layer(
            hmo_rec_m1, hvo_rec_m1, w, b, vb, t, d)

        hmi_rec, hvi_rec, hmo_rec, hvo_rec = recurrent_layer(
            hmi, hvi, r, t, j, d)

        exprs.update({
            'hidden_in_mean_%i' % (i + 1): hmi,
            'hidden_in_var_%i' % (i + 1): hvi,
            'hidden_mean_%i' % (i + 1): hmo,
            'hidden_var_%i' % (i + 1): hvo
        })

    output_in_mean, output_in_var, _, _ = forward_layer(
        hmo_rec, hvo_rec, hidden_to_out,
        out_bias, out_var_scale_sqrt,
        lambda x,y: (x, y), p_dropouts[-1])

    if in_to_out is not None:
        output_mean_inc, output_var_inc, _, _= forward_layer(
            inpt_mean, inpt_var, in_to_out,
            T.zeros_like(out_bias), T.ones_like(out_bias),
            lambda x, y: (x, y), p_dropouts[0])
        output_in_mean += output_mean_inc
        output_in_var += output_var_inc
    if skip_to_outs is not None:
        for i, s in enumerate(skip_to_outs):
            output_mean_inc, output_var_inc, _, _= forward_layer(
                exprs['hidden_mean_%i' % i], exprs['hidden_var_%i' % i], s,
                T.zeros_like(out_bias), T.ones_like(out_bias),
                lambda x, y: (x, y), p_dropouts[i + 1])
            output_in_mean += output_mean_inc
            output_in_var += output_var_inc

    output_mean, output_var = f_output(output_in_mean, output_in_var)

    # TODO: raise not implemented for out scale

    exprs.update({
        'inpt_mean': inpt_mean,
        'inpt_var': inpt_var,
        'output_in_mean': output_in_mean,
        'output_in_var': output_in_var,
        'output_mean': output_mean,
        'output_var': output_var,
        'output': T.concatenate([output_mean, output_var], axis=2),
    })

    return exprs
