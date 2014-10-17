# -*- coding: utf-8 -*-
"""Module implementing variance propagation and fast dropout for bidirectional
recurrent networks.
"""


import theano.tensor as T

from ...util import lookup
from ...component.varprop import transfer

from rnn import int_forward_layer, forward_layer, recurrent_layer


def parameters(n_inpt, n_hiddens, n_output, skip_to_out=False, prefix=''):
    spec = dict(in_to_hidden=(n_inpt, n_hiddens[0]),
                hidden_to_out=(n_hiddens[-1], n_output),
                hidden_bias_0=n_hiddens[0],
                out_bias=n_output)

    zipped = zip(n_hiddens[:-1], n_hiddens[1:])
    for i, (inlayer, outlayer) in enumerate(zipped):
        spec['hidden_to_hidden_%i' % i] = (inlayer, outlayer)

    if skip_to_out:
        spec['in_to_out'] = (n_inpt, n_output)

    for i, h in enumerate(n_hiddens):
        spec['hidden_bias_%i' % i] = h
        spec['recurrent_fwd_%i' % i] = (h, h)
        spec['recurrent_bwd_%i' % i] = (h, h)
        spec['initial_hidden_means_fwd_%i' % i] = h
        spec['initial_hidden_means_bwd_%i' % i] = h
        spec['initial_hidden_vars_fwd_%i' % i] = h
        spec['initial_hidden_vars_bwd_%i' % i] = h
        if skip_to_out and i < len(n_hiddens):
            # Only do for all but the last layer.
            spec['hidden_%i_to_out' % i] = (h, n_output)

    spec = dict(('%s%s' % (prefix, k), v) for k, v in spec.items())

    return spec


def exprs(inpt_mean, inpt_var, in_to_hidden, hidden_to_hiddens, hidden_to_out,
          hidden_biases, hidden_var_scales_sqrt,
          initial_hiddens_fwd, initial_hiddens_bwd,
          recurrents_fwd, recurrents_bwd,
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

    initial_hiddens_fwd : list of Theano variables
        List of vectors representing the initial hidden states for the forward
        network.

    initial_hiddens_bwd : list of Theano variables
        List of vectors representing the initial hidden states for the backward
        network.

    recurrents_fwd : list of Theano variables
        List of matrices representing the recurrent weight matrices for the
        forward network.

    recurrents_bwd : list of Theano variables
        List of matrices representing the recurrent weight matrices for the
        backward network.

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

    hmi_rec_fwd, hvi_rec_fwd, hmo_rec_fwd, hvo_rec_fwd = recurrent_layer(
        hmi, hvi, recurrents_fwd[0], f_hiddens[0], initial_hiddens_fwd[0],
        p_dropouts[1])

    hmi_rec_bwd, hvi_rec_bwd, hmo_rec_bwd, hvo_rec_bwd = recurrent_layer(
        hmi[::-1], hvi[::-1], recurrents_bwd[0], f_hiddens[0], initial_hiddens_bwd[0],
        p_dropouts[1])

    hmo_rec = (hmo_rec_fwd + hmo_rec_bwd[::-1]) / 2.
    hvo_rec = (hvo_rec_fwd + hvo_rec_bwd[::-1]) / 4.

    exprs.update({
        'hidden_in_mean_0_fwd': hmi_rec_fwd,
        'hidden_in_var_0_fwd': hvi_rec_fwd,
        'hidden_mean_0_fwd': hmo_rec_fwd,
        'hidden_var_0_fwd': hvo_rec_fwd,

        'hidden_in_mean_0_bwd': hmi_rec_bwd,
        'hidden_in_var_0_bwd': hvi_rec_bwd,
        'hidden_mean_0_bwd': hmo_rec_bwd,
        'hidden_var_0_bwd': hvo_rec_bwd,

        'hidden_mean_0': hmo_rec,
        'hidden_var_0': hvo_rec,
    })

    zipped = zip(
        hidden_to_hiddens, hidden_biases[1:], hidden_var_scales_sqrt[1:],
        recurrents_fwd[1:], recurrents_bwd[1:],
        f_hiddens[1:],
        initial_hiddens_fwd[1:], initial_hiddens_bwd[1:],
        p_dropouts[1:])

    for i, (w, b, vb, rf, rb, t, jf, jb, d) in enumerate(zipped):
        hmo_rec_m1, hvo_rec_m1 = hmo_rec, hvo_rec

        hmi, hvi, hmo, hvo = forward_layer(
            hmo_rec_m1, hvo_rec_m1, w, b, vb, t, d)

        hmi_rec_f, hvi_rec_f, hmo_rec_f, hvo_rec_f = recurrent_layer(
            hmi, hvi, rf, t, jf, d)

        hmi_rec_b, hvi_rec_b, hmo_rec_b, hvo_rec_b = recurrent_layer(
            hmi, hvi, rb, t, jb, d)

        hmo_rec = (hmo_rec_f + hmo_rec_b[::-1]) / 2.
        hvo_rec = (hvo_rec_f + hvo_rec_b[::-1]) / 4.

    output_in_mean, output_in_var, _, _ = forward_layer(
        hmo_rec, hvo_rec, hidden_to_out,
        out_bias, hidden_var_scales_sqrt[-1],
        lambda x, y: (x, y), p_dropouts[-1])

    if in_to_out is not None:
        output_mean_inc, output_var_inc, _, _ = forward_layer(
            inpt_mean, inpt_var, in_to_out,
            T.zeros_like(out_bias), T.ones_like(out_bias),
            lambda x, y: (x, y), p_dropouts[0])
        output_in_mean += output_mean_inc
        output_in_var += output_var_inc
    if skip_to_outs is not None:
        for i, s in enumerate(skip_to_outs):
            output_mean_inc, output_var_inc, _, _ = forward_layer(
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
