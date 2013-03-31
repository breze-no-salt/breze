# -*- coding: utf-8 -*-


import numpy as np

import theano
import theano.tensor as T
from theano.tensor.extra_ops import repeat
from theano.tensor.shared_randomstreams import RandomStreams

from ...util import ParameterSet, Model, lookup
from ...component import transfer, loss as loss_


def recurrent_layer(hidden_inpt, hidden_to_hidden, f, initial_hidden):
    def step(x, hi_tm1):
        h_tm1 = f(hi_tm1)
        hi = T.dot(h_tm1, hidden_to_hidden) + x
        return hi

    # Modify the initial hidden state to obtain several copies of
    # it, one per sample.
    initial_hidden_b = repeat(initial_hidden, hidden_inpt.shape[1], axis=0)
    initial_hidden_b = initial_hidden_b.reshape(
        (hidden_inpt.shape[1], hidden_inpt.shape[2]))

    hidden_in_rec, _ = theano.scan(
        step,
        sequences=hidden_inpt,
        outputs_info=[initial_hidden_b])

    hidden_rec = f(hidden_in_rec)

    return hidden_in_rec, hidden_rec


def lstm_layer(hidden_inpt, hidden_to_hidden,
               ingate_peephole, outgate_peephole, forgetgate_peephole,
               f):
    n_hidden_out = hidden_to_hidden.shape[0]

    def lstm_step(x_t, s_tm1, h_tm1):
        x_t += T.dot(h_tm1, hidden_to_hidden)

        inpt = T.tanh(x_t[:, :n_hidden_out])
        gates = x_t[:, n_hidden_out:]
        inpeep = s_tm1 * ingate_peephole
        outpeep = s_tm1 * outgate_peephole
        forgetpeep = s_tm1 * forgetgate_peephole

        ingate = f(gates[:, :n_hidden_out] + inpeep)
        forgetgate = f(
            gates[:, n_hidden_out:2 * n_hidden_out] + forgetpeep)
        outgate = f(gates[:, 2 * n_hidden_out:] + outpeep)

        s_t = inpt * ingate + s_tm1 * forgetgate
        h_t = f(s_t) * outgate
        return [s_t, h_t]

    (states, hidden_rec), _ = theano.scan(
        lstm_step,
        sequences=hidden_inpt,
        outputs_info=[T.zeros_like(hidden_inpt[0, :, 0:n_hidden_out]),
                      T.zeros_like(hidden_inpt[0, :, 0:n_hidden_out])
                      ])

    return states, hidden_rec


def feedforward_layer(inpt, weights, bias):
    n_time_steps = inpt.shape[0]
    n_samples = inpt.shape[1]

    n_inpt = weights.shape[0]
    n_output = weights.shape[1]

    inpt_flat = inpt.reshape((n_time_steps * n_samples, n_inpt))
    output_flat = T.dot(inpt_flat, weights)
    output = output_flat.reshape((n_time_steps, n_samples, n_output))
    output += bias.dimshuffle('x', 'x', 0)
    return output


def leaky_integration(inpt, coefficients):
    def step(x, y_tm1):
        c = coefficients[np.newaxis]
        y = c * y_tm1 + (1 - c) * x
        return y
    output, _ = theano.scan(
        step,
        sequences=inpt,
        outputs_info=[T.zeros_like(inpt[0])])
    return output


def multinomial_weights(inpt):
    # Numerical stability.
    inpt = T.maximum(inpt, -10)
    inpt_normed = inpt - inpt.min(axis=0).dimshuffle('x', 0)
    inpt_normed = T.minimum(inpt_normed, 100)
    return T.exp(inpt_normed) / (T.exp(inpt_normed).sum(axis=0) + 1e-4)


def weighted_pooling(inpt):
    # First do a stable softmax over time.
    inpt_flat = inpt.reshape((inpt.shape[0], inpt.shape[1] * inpt.shape[2]))
    p = multinomial_weights(inpt_flat)

    inpt_flat *= p
    res_flat = inpt_flat.sum(axis=0)
    return res_flat.reshape((inpt.shape[1], inpt.shape[2]))


def pooling_layer(inpt, typ):
    if typ == 'mean':
        output = T.mean(inpt, axis=0)
    elif typ == 'sum':
        output = T.sum(inpt, axis=0)
    elif typ == 'prod':
        output = T.prod(inpt, axis=0)
    elif typ == 'min':
        output = T.min(inpt, axis=0)
    elif typ == 'max':
        output = T.max(inpt, axis=0)
    elif typ == 'last':
        output = inpt[-1]
    elif typ == 'stochastic':
        output = stochastic_pooling(inpt)
    else:
        raise ValueError('unknown pooling operator %s' % typ)
    return output


def stochastic_pooling(inpt, rng=None):
    if rng is None:
        srng = RandomStreams()

    # First do a stable softmax over time.
    inpt_flat = inpt.reshape((inpt.shape[0], inpt.shape[1] * inpt.shape[2]))
    p = multinomial_weights(inpt_flat)

    # Sum up the probabilities giving the cdf.
    cumulative, _ = theano.scan(
        lambda prior_result, c: prior_result + c,
        p,
        outputs_info=T.zeros_like(p[0]))

    # Draw Uniformly and check into which interval of the cdf the sample falls.
    u = srng.uniform(size=inpt_flat.shape)[0, :]
    picks = T.eq((u < cumulative), 1)
    idxs = T.argmax(picks, axis=0)

    # Return that sample.
    res_flat = inpt_flat[idxs, T.arange(0, idxs.shape[0])]
    return res_flat.reshape((inpt.shape[1], inpt.shape[2]))


def rnn(inpt, in_to_hidden, hidden_to_hiddens, hidden_to_out,
        hidden_biases, initial_hiddens, recurrents, out_bias, hidden_transfers,
        out_transfer, pooling, leaky_coeffs=None):
        exprs = {}

        f_hiddens = [lookup(i, transfer) for i in hidden_transfers]
        f_output = lookup(out_transfer, transfer)

        hidden_in = feedforward_layer(inpt, in_to_hidden, hidden_biases[0])
        hidden_in_rec, hidden_rec = recurrent_layer(
            hidden_in, recurrents[0], f_hiddens[0], initial_hiddens[0])
        exprs['hidden_in_0'] = hidden_in_rec
        if leaky_coeffs is not None:
            hidden_rec = leaky_integration(hidden_rec, leaky_coeffs[0])
        exprs['hidden_0'] = hidden_rec

        zipped = zip(hidden_to_hiddens, hidden_biases[1:], recurrents[1:],
                     f_hiddens[1:], initial_hiddens[1:])

        for i, (w, b, r, t, j) in enumerate(zipped):
            hidden_m1 = hidden_rec
            hidden_in = feedforward_layer(hidden_m1, w, b)
            hidden_in_rec, hidden_rec = recurrent_layer(hidden_in, r, t, j)
            if leaky_coeffs is not None:
                hidden_rec = leaky_integration(hidden_rec, leaky_coeffs[i])
            exprs['hidden_in_%i' % (i + 1)] = hidden_in_rec
            exprs['hidden_%i' % (i + 1)] = hidden_rec

        unpooled = feedforward_layer(hidden_rec, hidden_to_out, out_bias)

        if pooling is None:
            output_in = unpooled
        else:
            output_in = pooling_layer(unpooled, pooling)

        output = f_output(output_in)

        exprs.update(
            {'inpt': inpt,
             'unpooled': unpooled,
             'output_in': output_in,
             'output': output,
             })

        return exprs


def lstm_rnn(inpt, in_to_hidden, hidden_to_hiddens, hidden_to_out,
             hidden_biases, recurrents, out_bias,
             ingate_peepholes, outgate_peepholes, forgetgate_peepholes,
             hidden_transfers, out_transfer, pooling, leaky_coeffs=None):
        exprs = {}

        f_hiddens = [lookup(i, transfer) for i in hidden_transfers]
        f_output = lookup(out_transfer, transfer)

        # First ordinary feedforward layer.
        hidden_in = feedforward_layer(inpt, in_to_hidden, hidden_biases[0])

        # First recurrent layer.
        state, hidden_rec = lstm_layer(
            hidden_in, recurrents[0],
            ingate_peepholes[0], outgate_peepholes[0], forgetgate_peepholes[0],
            f_hiddens[0])

        exprs['state_0'] = state
        exprs['hidden_0'] = hidden_rec

        if leaky_coeffs is not None:
            hidden_rec = leaky_integration(hidden_rec, leaky_coeffs[0])

        exprs['hidden_0'] = hidden_rec

        # Optional further recurrent layers.
        zipped = zip(hidden_to_hiddens, hidden_biases[1:], recurrents[1:],
                     ingate_peepholes[1:], outgate_peepholes[1:],
                     forgetgate_peepholes[1:],
                     f_hiddens[1:])
        for i, (w, b, r, ig, og, fg, t) in enumerate(zipped):
            hidden_m1 = hidden_rec
            hidden_in = feedforward_layer(hidden_m1, w, b)

            state, hidden_rec = lstm_layer(hidden_in, r, ig, og, fg, t)

            if leaky_coeffs is not None:
                hidden_rec = leaky_integration(hidden_rec, leaky_coeffs[i])

            exprs['state_%i' % (i + 1)] = state
            exprs['hidden_%i' % (i + 1)] = hidden_rec

        unpooled = feedforward_layer(hidden_rec, hidden_to_out, out_bias)

        if pooling is None:
            output_in = unpooled
        else:
            output_in = pooling_layer(unpooled, pooling)

        output = f_output(output_in)

        exprs.update(
            {'inpt': inpt,
             'unpooled': unpooled,
             'output_in': output_in,
             'output': output,
             })

        return exprs


class BaseRecurrentNetwork(Model):

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer='identity', loss='squared',
                 pooling=None, leaky_coeffs=None):
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
        self.pooling = pooling
        self.leaky_coeffs = leaky_coeffs
        super(BaseRecurrentNetwork, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(
            self.n_inpt, self.n_hiddens, self.n_output)
        self.parameters = ParameterSet(**parspec)


class LstmNetworkComponent(object):

    @staticmethod
    def get_parameter_spec(n_inpt, n_hiddens, n_output):
        spec = {
            'in_to_hidden': (n_inpt, 4 * n_hiddens[0]),
            'hidden_to_out': (n_hiddens[-1], n_output),
            'hidden_bias_0': 4 * n_hiddens[0],
            'recurrent_0': (n_hiddens[0], 4 * n_hiddens[0]),
            'out_bias': n_output,
            'ingate_peephole_0': (n_hiddens[0],),
            'outgate_peephole_0': (n_hiddens[0],),
            'forgetgate_peephole_0': (n_hiddens[0],)
        }

        zipped = zip(n_hiddens[:-1], n_hiddens[1:])
        for i, (inlayer, outlayer) in enumerate(zipped):
            spec.update({
                'hidden_bias_%i' % (i + 1): 4 * outlayer,
                'hidden_to_hidden_%i': (inlayer, 4 * outlayer),
                'recurrent_%i': (outlayer, 4 * outlayer),
                'ingate_peephole_%i' % (i + 1): (outlayer,),
                'outgate_peephole_%i' % (i + 1): (outlayer,),
                'forgetgate_peephole_%i' % (i + 1): (outlayer,)
            })
        return spec


class SimpleRnnComponent(object):

    @staticmethod
    def get_parameter_spec(n_inpt, n_hiddens, n_output):
        spec = {
            'in_to_hidden': (n_inpt, n_hiddens[0]),
            'hidden_bias_0': n_hiddens[0],
            'recurrent_0': (n_hiddens[0], n_hiddens[0]),
            'hidden_to_out': (n_hiddens[-1], n_output),
            'initial_hidden_0': n_hiddens[0],
            'out_bias': n_output
        }

        zipped = zip(n_hiddens[:-1], n_hiddens[1:])
        for i, (inlayer, outlayer) in enumerate(zipped):
            spec['hidden_to_hidden_%i' % i] = (inlayer, outlayer)
            spec['hidden_bias_%i' % (i + 1)] = outlayer
            spec['recurrent_%i' % (i + 1)] = (n_hiddens[i], n_hiddens[i])
            spec['initial_hidden_%i' % (i + 1)] = outlayer

        return spec


class UnsupervisedRecurrentNetwork(BaseRecurrentNetwork, SimpleRnnComponent):

    def init_exprs(self):
        inpt = T.tensor3('inpt')
        pars = self.parameters
        hidden_to_hiddens = [getattr(pars, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(pars, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        recurrents = [getattr(pars, 'recurrent_%i' % i)
                      for i in range(len(self.n_hiddens))]
        initial_hiddens = [getattr(pars, 'initial_hidden_%i' % i)
                           for i in range(len(self.n_hiddens))]
        self.exprs = self.make_exprs(
            inpt,
            pars.in_to_hidden, hidden_to_hiddens, pars.hidden_to_out,
            hidden_biases, initial_hiddens, recurrents, pars.out_bias,
            self.hidden_transfers, self.out_transfer, self.loss,
            self.pooling)

    @staticmethod
    def make_exprs(inpt, in_to_hidden, hidden_to_hiddens, hidden_to_out,
                   hidden_biases, initial_hiddens, recurrents, out_bias,
                   hidden_transfers, out_transfer, loss, pooling):
        exprs = rnn(inpt, in_to_hidden, hidden_to_hiddens,
                    hidden_to_out, hidden_biases, initial_hiddens, recurrents,
                    out_bias, hidden_transfers, out_transfer, pooling)
        f_loss = lookup(loss, loss_)
        loss = f_loss(exprs['output'])

        # We need to check whether the loss is a scalar. If it is not,
        # we get a row wise and component wise loss, which we sum away
        # component wise and mean away row wise.
        if loss.ndim != 0:
            sum_axis = 2 if not pooling else 1
            loss_row_wise = loss.sum(axis=sum_axis)
            exprs['loss_row_wise'] = loss_row_wise

            loss = loss_row_wise.mean()

        exprs['loss'] = loss
        return exprs


class SupervisedRecurrentNetwork(BaseRecurrentNetwork, SimpleRnnComponent):

    def init_exprs(self):
        inpt = T.tensor3('inpt')
        pars = self.parameters

        hidden_to_hiddens = [getattr(pars, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(pars, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        recurrents = [getattr(pars, 'recurrent_%i' % i)
                      for i in range(len(self.n_hiddens))]
        initial_hiddens = [getattr(pars, 'initial_hidden_%i' % i)
                           for i in range(len(self.n_hiddens))]

        if self.pooling is None:
            target = T.tensor3('target')
        else:
            target = T.matrix('target')

        self.exprs = self.make_exprs(
            inpt, target,
            pars.in_to_hidden, hidden_to_hiddens, pars.hidden_to_out,
            hidden_biases, initial_hiddens, recurrents, pars.out_bias,
            self.hidden_transfers, self.out_transfer, self.loss,
            self.pooling, self.leaky_coeffs)

    @staticmethod
    def make_exprs(inpt, target, in_to_hidden, hidden_to_hiddens, hidden_to_out,
                   hidden_biases, initial_hiddens, recurrents, out_bias,
                   hidden_transfers, out_transfer, loss, pooling, leaky_coeffs):
        exprs = rnn(inpt, in_to_hidden, hidden_to_hiddens,
                    hidden_to_out, hidden_biases, initial_hiddens, recurrents,
                    out_bias, hidden_transfers, out_transfer, pooling,
                    leaky_coeffs)
        f_loss = lookup(loss, loss_)
        sum_axis = 2 if not pooling else 1
        loss_row_wise = f_loss(target, exprs['output']).sum(axis=sum_axis)
        loss = loss_row_wise.mean()
        exprs['target'] = target
        exprs['loss'] = loss
        exprs['loss_row_wise'] = loss_row_wise
        return exprs


class UnsupervisedLstmRecurrentNetwork(BaseRecurrentNetwork, LstmNetworkComponent):

    def init_exprs(self):
        inpt = T.tensor3('inpt')
        pars = self.parameters

        hidden_to_hiddens = [getattr(pars, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(pars, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        recurrents = [getattr(pars, 'recurrent_%i' % i)
                      for i in range(len(self.n_hiddens))]
        ingate_peepholes = [getattr(pars, 'ingate_peephole_%i' % i)
                            for i in range(len(self.n_hiddens))]
        outgate_peepholes = [getattr(pars, 'outgate_peephole_%i' % i)
                             for i in range(len(self.n_hiddens))]
        forgetgate_peepholes = [getattr(pars, 'forgetgate_peephole_%i' % i)
                                for i in range(len(self.n_hiddens))]

        self.exprs = self.make_exprs(
            inpt,
            pars.in_to_hidden, hidden_to_hiddens, pars.hidden_to_out,
            hidden_biases, recurrents, pars.out_bias,
            ingate_peepholes, outgate_peepholes, forgetgate_peepholes,
            self.hidden_transfers, self.out_transfer, self.loss, self.pooling,
            self.leaky_coeffs)

    @staticmethod
    def make_exprs(inpt,
                   in_to_hidden, hidden_to_hiddens, hidden_to_out,
                   hidden_biases, recurrents, out_bias,
                   ingate_peepholes, outgate_peepholes, forgetgate_peepholes,
                   hidden_transfers, out_transfer, loss, pooling, leaky_coeffs):

        exprs = lstm_rnn(
            inpt, in_to_hidden, hidden_to_hiddens, hidden_to_out,
            hidden_biases, recurrents, out_bias,
            ingate_peepholes, outgate_peepholes, forgetgate_peepholes,
            hidden_transfers, out_transfer, pooling, leaky_coeffs)

        f_loss = lookup(loss, loss_)
        sum_axis = 2 if not pooling else 1
        loss = f_loss(exprs['output']).sum(axis=sum_axis).mean()
        exprs['loss'] = loss
        return exprs


class SupervisedLstmRecurrentNetwork(BaseRecurrentNetwork, LstmNetworkComponent):

    def init_exprs(self):
        inpt = T.tensor3('inpt')
        if self.pooling is None:
            target = T.tensor3('target')
        else:
            target = T.matrix('tensor3')
        pars = self.parameters

        hidden_to_hiddens = [getattr(pars, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(pars, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        recurrents = [getattr(pars, 'recurrent_%i' % i)
                      for i in range(len(self.n_hiddens))]
        ingate_peepholes = [getattr(pars, 'ingate_peephole_%i' % i)
                            for i in range(len(self.n_hiddens))]
        outgate_peepholes = [getattr(pars, 'outgate_peephole_%i' % i)
                             for i in range(len(self.n_hiddens))]
        forgetgate_peepholes = [getattr(pars, 'forgetgate_peephole_%i' % i)
                                for i in range(len(self.n_hiddens))]

        self.exprs = self.make_exprs(
            inpt, target,
            pars.in_to_hidden, hidden_to_hiddens, pars.hidden_to_out,
            hidden_biases, recurrents, pars.out_bias,
            ingate_peepholes, outgate_peepholes, forgetgate_peepholes,
            self.hidden_transfers, self.out_transfer, self.loss, self.pooling,
            self.leaky_coeffs)

    @staticmethod
    def make_exprs(inpt, target,
                   in_to_hidden, hidden_to_hiddens, hidden_to_out,
                   hidden_biases, recurrents, out_bias,
                   ingate_peepholes, outgate_peepholes, forgetgate_peepholes,
                   hidden_transfers, out_transfer, loss, pooling, leaky_coeffs):

        exprs = lstm_rnn(
            inpt, in_to_hidden, hidden_to_hiddens, hidden_to_out,
            hidden_biases, recurrents, out_bias,
            ingate_peepholes, outgate_peepholes, forgetgate_peepholes,
            hidden_transfers, out_transfer, pooling, leaky_coeffs)

        f_loss = lookup(loss, loss_)
        sum_axis = 2 if not pooling else 1
        loss = f_loss(target, exprs['output']).sum(axis=sum_axis).mean()

        exprs['loss'] = loss
        exprs['target'] = target

        return exprs
