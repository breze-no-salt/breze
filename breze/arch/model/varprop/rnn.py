# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
from theano.tensor.extra_ops import repeat

from ...util import lookup
from ...component.varprop import transfer, loss as loss_
from ...model.sequential.rnn import BaseRecurrentNetwork, SimpleRnnComponent
import mlp


def flat_time(x):
    return x.reshape((x.shape[0] * x.shape[1], x.shape[2]))


def unflat_time(x, time_steps):
    return x.reshape((time_steps, x.shape[0] // time_steps, x.shape[1]))


def forward_layer(in_mean, in_var, weights, mean_bias, var_bias_sqrt,
                  f, p_dropout):
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
    def step(inpt_mean, inpt_var, him_m1, hiv_m1):
        hom_m1, hov_m1 = f(him_m1, hiv_m1)
        hom = T.dot(hom_m1, weights) * p_dropout + inpt_mean
        dropout_var = p_dropout * (1 - p_dropout)
        hov = (T.dot(him_m1 ** 2, weights ** 2) * dropout_var
               + T.dot(hiv_m1, weights ** 2) * p_dropout
               + T.dot(hiv_m1, weights ** 2) * dropout_var
               + inpt_var)
        return hom, hov

    initial_hidden_mean = repeat(initial_hidden, in_mean.shape[1], axis=0)
    initial_hidden_mean = initial_hidden_mean.reshape(
        (in_mean.shape[1], in_mean.shape[2]))

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
        p_dropouts[0])

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
        out_bias, T.zeros_like(out_bias) + 1e-8,
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
