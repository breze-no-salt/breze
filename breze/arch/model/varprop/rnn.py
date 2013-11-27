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
from ...model.sequential.rnn import BaseRecurrentNetwork, SimpleRnnComponent
import mlp


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
    def step(inpt_mean, inpt_var, him_m1, hiv_m1):
        hom_m1, hov_m1 = f(him_m1, hiv_m1)
        hom = T.dot(hom_m1, weights) * p_dropout + inpt_mean

        p_keep = 1 - p_dropout
        dropout_var = p_dropout * (1 - p_dropout)

        element_var = (hov_m1 * dropout_var
                       + (hom_m1 ** 2) * dropout_var
                       + hov_m1 * p_keep ** 2)

        hov = T.dot(element_var, weights ** 2) + inpt_var

        return hom, hov

    initial_hidden_mean = repeat(initial_hidden.dimshuffle('x', 0), in_mean.shape[1], axis=0)

    initial_hidden_var = T.zeros_like(initial_hidden_mean) + 1e-8

    (hidden_in_mean_rec, hidden_in_var_rec), _ = theano.scan(
        step,
        sequences=[in_mean, in_var],
        outputs_info=[initial_hidden_mean, initial_hidden_var])

    hidden_mean_rec, hidden_var_rec = f(
        hidden_in_mean_rec, hidden_in_var_rec)

    return (hidden_in_mean_rec, hidden_in_var_rec,
            hidden_mean_rec, hidden_var_rec)


def rnn(inpt_mean, inpt_var, in_to_hidden, hidden_to_hiddens, hidden_to_out,
        hidden_biases, hidden_var_biases_sqrt, initial_hiddens, recurrents,
        out_bias, hidden_transfers, out_transfer, p_dropouts):
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

    hidden_var_biases_sqrt : Theano variable
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
    # TODO: add pooling
    # TODO: add leaky integration
    exprs = {}

    f_hiddens = [lookup(i, transfer) for i in hidden_transfers]
    f_output = lookup(out_transfer, transfer)

    hmi, hvi, hmo, hvo = forward_layer(
        inpt_mean, inpt_var, in_to_hidden,
        hidden_biases[0], hidden_var_biases_sqrt[0],
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
        hidden_to_hiddens, hidden_biases[1:], hidden_var_biases_sqrt[1:],
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

    output_in_mean, output_in_var, output_mean, output_var = forward_layer(
        hmo_rec, hvo_rec, hidden_to_out,
        out_bias, T.ones_like(out_bias),
        f_output, p_dropouts[-1])

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


class SupervisedRecurrentNetwork(BaseRecurrentNetwork, SimpleRnnComponent):

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer='identity', loss='squared',
                 pooling=None, leaky_coeffs=None,
                 p_dropout_inpt=.2, p_dropout_hidden=.5):
        self.n_inpt = n_inpt
        self.n_output = n_output

        # If these are not lists, we implicitly assume that we are dealing
        # with a single hidden layer architecture.
        self.n_hiddens = (
            n_hiddens if isinstance(n_hiddens, (list, tuple))
            else [n_hiddens])
        self.hidden_transfers = (
            hidden_transfers if isinstance(hidden_transfers, (list, tuple))
            else [hidden_transfers])

        self.out_transfer = out_transfer
        self.loss = loss

        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hidden = p_dropout_hidden

        super(SupervisedRecurrentNetwork, self).__init__(
            n_inpt, n_hiddens, n_output,
            hidden_transfers, out_transfer, loss, pooling, leaky_coeffs)

    def init_exprs(self):
        inpt_mean = T.tensor3('inpt_mean')
        inpt_var = T.tensor3('inpt_var')
        target = T.tensor3('target')
        pars = self.parameters

        hidden_to_hiddens = [getattr(pars, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(pars, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        # TODO: make real variance biases
        hidden_var_biases_sqrt = [T.zeros_like(i) + 1e-8 for i in hidden_biases]
        recurrents = [getattr(pars, 'recurrent_%i' % i)
                      for i in range(len(self.n_hiddens))]
        initial_hiddens = [getattr(pars, 'initial_hidden_%i' % i)
                           for i in range(len(self.n_hiddens))]

        self.exprs = self.make_exprs(
            inpt_mean, inpt_var, target,
            pars.in_to_hidden, hidden_to_hiddens, pars.hidden_to_out,
            hidden_biases, hidden_var_biases_sqrt,
            initial_hiddens, recurrents, pars.out_bias,
            self.hidden_transfers, self.out_transfer, self.loss,
            self.pooling, self.leaky_coeffs,
            [self.p_dropout_inpt] + [self.p_dropout_hidden] * len(recurrents))

    @staticmethod
    def make_exprs(inpt_mean, inpt_var, target, in_to_hidden, hidden_to_hiddens, hidden_to_out,
                   hidden_biases, hidden_var_biases_sqrt,
                   initial_hiddens, recurrents, out_bias,
                   hidden_transfers, out_transfer, loss, pooling, leaky_coeffs,
                   p_dropouts):
        if pooling is not None:
            raise NotImplementedError("I don't know about pooling yet.")
        if leaky_coeffs is not None:
            raise NotImplementedError("I don't know about leaky coefficiens "
                                      "yet.")
        exprs = rnn(inpt_mean, inpt_var, in_to_hidden, hidden_to_hiddens,
                    hidden_to_out, hidden_biases, hidden_var_biases_sqrt,
                    initial_hiddens, recurrents,
                    out_bias, hidden_transfers, out_transfer, p_dropouts)
        f_loss = lookup(loss, loss_)
        sum_axis = 2
        loss_row_wise = f_loss(target, exprs['output']).sum(axis=sum_axis)
        loss = loss_row_wise.mean()
        exprs['target'] = target
        exprs['loss'] = loss
        exprs['loss_row_wise'] = loss_row_wise
        return exprs


class FastDropoutRnn(SupervisedRecurrentNetwork):

    inpt_var = 0

    def init_exprs(self):
        inpt_mean = T.tensor3('inpt_mean')
        inpt_var = T.ones_like(inpt_mean) * self.inpt_var
        target = T.tensor3('target')
        pars = self.parameters

        hidden_to_hiddens = [getattr(self.parameters, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(self.parameters, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        hidden_var_biases_sqrt = [1 for i in hidden_biases]
        recurrents = [getattr(pars, 'recurrent_%i' % i)
                      for i in range(len(self.n_hiddens))]
        initial_hiddens = [getattr(pars, 'initial_hidden_%i' % i)
                           for i in range(len(self.n_hiddens))]

        self.exprs = self.make_exprs(
            inpt_mean, inpt_var, target,
            pars.in_to_hidden, hidden_to_hiddens, pars.hidden_to_out,
            hidden_biases,
            hidden_var_biases_sqrt,
            initial_hiddens, recurrents, pars.out_bias,
            self.hidden_transfers, self.out_transfer, self.loss,
            self.pooling, self.leaky_coeffs,
            [self.p_dropout_inpt] + [self.p_dropout_hidden] * len(recurrents))

        self.exprs['inpt'] = inpt_mean
