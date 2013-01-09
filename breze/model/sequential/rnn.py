# -*- coding: utf-8 -*-


import theano
import theano.tensor as T

from ...util import ParameterSet, Model, lookup
from ...component import transfer, loss as loss_


def recurrent_layer(hidden_inpt, hidden_to_hidden, f):
    def step(x, hi_tm1):
        h_tm1 = f(hi_tm1)
        hi = T.dot(h_tm1, hidden_to_hidden) + x
        return hi

    hidden_in_rec, _ = theano.scan(
        step,
        sequences=hidden_inpt,
        outputs_info=[T.zeros_like(hidden_inpt[0])])

    hidden_rec = f(hidden_in_rec)

    return hidden_in_rec, hidden_rec


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


def rnn(inpt, in_to_hidden, hidden_to_hiddens, hidden_to_out,
        hidden_biases, recurrents, out_bias, hidden_transfers,
        out_transfer, pooling):
        exprs = {}

        f_hiddens = [lookup(i, transfer) for i in hidden_transfers]
        f_output = lookup(out_transfer, transfer)

        hidden_in = feedforward_layer(inpt, in_to_hidden, hidden_biases[0])
        hidden_in_rec, hidden_rec = recurrent_layer(
            hidden_in, recurrents[0], f_hiddens[0])
        exprs['hidden_in_0'] = hidden_in_rec
        exprs['hidden_0'] = hidden_rec

        zipped = zip(hidden_to_hiddens, hidden_biases[1:], recurrents[1:],
                     f_hiddens[1:])
        for i, (w, b, r, t) in enumerate(zipped):
            hidden_m1 = hidden_rec
            hidden_in = feedforward_layer(hidden_m1, w, b)
            hidden_in_rec, hidden_rec = recurrent_layer(hidden_in, r, t)
            exprs['hidden_in_%i' % (i + 1)] = hidden_in_rec
            exprs['hidden_%i' % (i + 1)] = hidden_rec

        output_in = feedforward_layer(hidden_rec, hidden_to_out, out_bias)

        if pooling is None:
            pass
        elif pooling == 'mean':
            output_in = T.mean(output_in, axis=0)
        elif pooling == 'sum':
            output_in = T.sum(output_in, axis=0)
        elif pooling == 'prod':
            output_in = T.prod(output_in, axis=0)
        elif pooling == 'min':
            output_in = T.min(output_in, axis=0)
        elif pooling == 'max':
            output_in = T.max(output_in, axis=0)
        else:
            raise ValueError('unknown pooling operator %s' % pooling)

        output = f_output(output_in)

        exprs.update(
            {'inpt': inpt,
             'output_in': output_in,
             'output': output,
             })

        return exprs


class BaseRecurrentNetwork(Model):

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer='identity', loss='squared',
                 pooling=None):
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
        super(BaseRecurrentNetwork, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(
            self.n_inpt, self.n_hiddens, self.n_output)
        self.parameters = ParameterSet(**parspec)

    @staticmethod
    def get_parameter_spec(n_inpt, n_hiddens, n_output):
        spec = {
            'in_to_hidden': (n_inpt, n_hiddens[0]),
            'hidden_bias_0': n_hiddens[0],
            'recurrent_0': (n_hiddens[0], n_hiddens[0]),
            'hidden_to_out': (n_hiddens[-1], n_output),
            'out_bias': n_output
        }

        zipped = zip(n_hiddens[:-1], n_hiddens[1:])
        for i, (inlayer, outlayer) in enumerate(zipped):
            spec['hidden_to_hidden_%i' % i] = (inlayer, outlayer)
            spec['hidden_bias_%i' % (i + 1)] = outlayer
            spec['recurrent_%i' % (i + 1)] = (n_hiddens[i], n_hiddens[i])

        return spec


class SupervisedRecurrentNetwork(BaseRecurrentNetwork):

    def init_exprs(self):
        inpt = T.tensor3('inpt')
        pars = self.parameters

        hidden_to_hiddens = [getattr(pars, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(pars, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        recurrents = [getattr(pars, 'recurrent_%i' % i)
                      for i in range(len(self.n_hiddens))]

        if self.pooling is None:
            target = T.tensor3('target')
        else:
            target = T.matrix('target')

        self.exprs = self.make_exprs(
            inpt, target,
            pars.in_to_hidden, hidden_to_hiddens, pars.hidden_to_out,
            hidden_biases, recurrents, pars.out_bias,
            self.hidden_transfers, self.out_transfer, self.loss,
            self.pooling)


    @staticmethod
    def make_exprs(inpt, target, in_to_hidden, hidden_to_hiddens, hidden_to_out,
                   hidden_biases, recurrents, out_bias, hidden_transfers,
                   out_transfer, loss, pooling):
        exprs = rnn(inpt, in_to_hidden, hidden_to_hiddens,
                    hidden_to_out, hidden_biases, recurrents, out_bias,
                    hidden_transfers, out_transfer, pooling)
        f_loss = lookup(loss, loss_)
        loss = f_loss(target, exprs['output']).sum(axis=2).mean()
        exprs['target'] = target
        exprs['loss'] = loss
        return exprs


class UnsupervisedRecurrentNetwork(BaseRecurrentNetwork):

    def init_exprs(self):
        inpt = T.tensor3('inpt')
        pars = self.parameters
        hidden_to_hiddens = [getattr(pars, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(pars, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        recurrents = [getattr(pars, 'recurrent_%i' % i)
                      for i in range(len(self.n_hiddens))]
        self.exprs = self.make_exprs(
            inpt,
            pars.in_to_hidden, hidden_to_hiddens, pars.hidden_to_out,
            hidden_biases, recurrents, pars.out_bias,
            self.hidden_transfers, self.out_transfer, self.loss,
            self.pooling)

    @staticmethod
    def make_exprs(inpt, in_to_hidden, hidden_to_hiddens, hidden_to_out,
                   hidden_biases, recurrents, out_bias, hidden_transfers,
                   out_transfer, loss, pooling):
        exprs = rnn(inpt, in_to_hidden, hidden_to_hiddens,
                    hidden_to_out, hidden_biases, recurrents, out_bias,
                    hidden_transfers, out_transfer, pooling)
        f_loss = lookup(loss, loss_)
        loss = f_loss(exprs['output']).sum(axis=2).mean()
        exprs['loss'] = loss
        return exprs


class SupervisedLstmRecurrentNetwork(SupervisedRecurrentNetwork):

    def __init__(self, n_inpt, n_hidden, n_output,
                 hidden_transfer, out_transfer='identity', loss='squared',
                 pooling=None):
        super(SupervisedLstmRecurrentNetwork, self).__init__(
            n_inpt, n_hidden, n_output, hidden_transfer, out_transfer, loss,
            pooling)

    @staticmethod
    def get_parameter_spec(n_inpt, n_hidden, n_output):
        return {
            'in_to_hidden': (n_inpt, 4 * n_hidden),
            'hidden_to_hidden': (n_hidden, 4 * n_hidden),
            'hidden_to_out': (n_hidden, n_output),
            'hidden_bias': 4 * n_hidden,
            'out_bias': n_output,
            'ingate_peephole': (n_hidden,),
            'outgate_peephole': (n_hidden,),
            'forgetgate_peephole': (n_hidden,)
        }

    def init_exprs(self):
        inpt = T.tensor3('inpt')
        if self.pooling is None:
            target = T.tensor3('target')
        else:
            target = T.matrix('tensor3')
        pars = self.parameters
        self.exprs = self.make_exprs(
            inpt, target,
            pars.in_to_hidden, pars.hidden_to_hidden, pars.hidden_to_out,
            pars.hidden_bias, pars.out_bias,
            pars.ingate_peephole, pars.outgate_peephole,
            pars.forgetgate_peephole,
            self.hidden_transfer, self.out_transfer, self.loss, self.pooling)

    @staticmethod
    def make_exprs(inpt, target,
                   in_to_hidden, hidden_to_hidden, hidden_to_out,
                   hidden_bias, out_bias,
                   ingate_peephole, outgate_peephole, forgetgate_peephole,
                   hidden_transfer, out_transfer, loss, pooling):

        f_hidden = lookup(hidden_transfer, transfer)
        f_output = lookup(out_transfer, transfer)
        f_loss = lookup(loss, loss_)

        n_time_steps = inpt.shape[0]
        n_samples = inpt.shape[1]
        n_inpt = in_to_hidden.shape[0]
        n_hidden_in = hidden_to_hidden.shape[1]
        n_hidden_out = hidden_to_hidden.shape[0]
        n_output = hidden_to_out.shape[1]

        # If we ever want to enable disabling of peepholes, we can make this
        # variable be setable to False.
        peepholes = True

        def lstm_step(x_t, s_tm1, h_tm1):
            x_t += T.dot(h_tm1, hidden_to_hidden)

            inpt = T.tanh(x_t[:, :n_hidden_out])
            gates = x_t[:, n_hidden_out:]
            inpeep = 0 if not peepholes else s_tm1 * ingate_peephole
            outpeep = 0 if not peepholes else s_tm1 * outgate_peephole
            forgetpeep = 0 if not peepholes else s_tm1 * forgetgate_peephole

            ingate = f_hidden(gates[:, :n_hidden_out] + inpeep)
            forgetgate = f_hidden(
                gates[:, n_hidden_out:2 * n_hidden_out] + forgetpeep)
            outgate = f_hidden(gates[:, 2 * n_hidden_out:] + outpeep)

            s_t = inpt * ingate + s_tm1 * forgetgate
            h_t = f_hidden(s_t) * outgate
            return [s_t, h_t]

        inpt_flat = inpt.reshape((n_time_steps * n_samples, n_inpt))
        hidden_flat = T.dot(inpt_flat, in_to_hidden)
        hidden = hidden_flat.reshape((n_time_steps, n_samples, n_hidden_in))
        hidden += hidden_bias.dimshuffle('x', 'x', 0)

        (states, hidden_rec), _ = theano.scan(
            lstm_step,
            sequences=hidden,
            outputs_info=[T.zeros_like(hidden[0, :, 0:n_hidden_out]),
                          T.zeros_like(hidden[0, :, 0:n_hidden_out])
                          ])

        hidden_rec_flat = hidden_rec.reshape(
            (n_time_steps * n_samples, n_hidden_out))

        output_flat = T.dot(hidden_rec_flat, hidden_to_out)
        output_in = output_flat.reshape((n_time_steps, n_samples, n_output))
        output_in += out_bias.dimshuffle('x', 'x', 0)

        if pooling is None:
            pass
        elif pooling == 'mean':
            output_in = T.mean(output_in, axis=0)
        elif pooling == 'sum':
            output_in = T.sum(output_in, axis=0)
        elif pooling == 'prod':
            output_in = T.prod(output_in, axis=0)
        elif pooling == 'min':
            output_in = T.min(output_in, axis=0)
        elif pooling == 'max':
            output_in = T.max(output_in, axis=0)
        else:
            raise ValueError('unknown pooling operator %s' % pooling)

        output = f_output(output_in)

        loss = f_loss(target, output).sum(axis=2).mean()

        return {'inpt': inpt,
                'target': target,
                'states': states,
                'hidden': hidden_rec,
                'output-in': output_in,
                'output': output,
                'loss': loss}


class UnsupervisedLstmRecurrentNetwork(UnsupervisedRecurrentNetwork):

    def __init__(self, n_inpt, n_hidden, n_output,
                 hidden_transfer, out_transfer='identity', loss='squared',
                 pooling=None):
        super(UnsupervisedLstmRecurrentNetwork, self).__init__(
            n_inpt, n_hidden, n_output, hidden_transfer, out_transfer, loss,
            pooling)

    @staticmethod
    def get_parameter_spec(n_inpt, n_hidden, n_output):
        return {
            'in_to_hidden': (n_inpt, 4 * n_hidden),
            'hidden_to_hidden': (n_hidden, 4 * n_hidden),
            'hidden_to_out': (n_hidden, n_output),
            'hidden_bias': 4 * n_hidden,
            'out_bias': n_output,
            'ingate_peephole': (n_hidden,),
            'outgate_peephole': (n_hidden,),
            'forgetgate_peephole': (n_hidden,)
        }

    def init_exprs(self):
        inpt = T.tensor3('inpt')
        pars = self.parameters
        self.exprs = self.make_exprs(
            inpt,
            pars.in_to_hidden, pars.hidden_to_hidden, pars.hidden_to_out,
            pars.hidden_bias, pars.out_bias,
            pars.ingate_peephole, pars.outgate_peephole,
            pars.forgetgate_peephole,
            self.hidden_transfer, self.out_transfer, self.loss, self.pooling)

    @staticmethod
    def make_exprs(inpt,
                   in_to_hidden, hidden_to_hidden, hidden_to_out,
                   hidden_bias, out_bias,
                   ingate_peephole, outgate_peephole, forgetgate_peephole,
                   hidden_transfer, out_transfer, loss, pooling):

        f_hidden = lookup(hidden_transfer, transfer)
        f_output = lookup(out_transfer, transfer)
        f_loss = lookup(loss, loss_)

        n_time_steps = inpt.shape[0]
        n_samples = inpt.shape[1]
        n_inpt = in_to_hidden.shape[0]
        n_hidden_in = hidden_to_hidden.shape[1]
        n_hidden_out = hidden_to_hidden.shape[0]
        n_output = hidden_to_out.shape[1]

        # If we ever want to enable disabling of peepholes, we can make this
        # variable be setable to False.
        peepholes = True

        def lstm_step(x_t, s_tm1, h_tm1):
            x_t += T.dot(h_tm1, hidden_to_hidden)

            inpt = T.tanh(x_t[:, :n_hidden_out])
            gates = x_t[:, n_hidden_out:]
            inpeep = 0 if not peepholes else s_tm1 * ingate_peephole
            outpeep = 0 if not peepholes else s_tm1 * outgate_peephole
            forgetpeep = 0 if not peepholes else s_tm1 * forgetgate_peephole

            ingate = f_hidden(gates[:, :n_hidden_out] + inpeep)
            forgetgate = f_hidden(
                gates[:, n_hidden_out:2 * n_hidden_out] + forgetpeep)
            outgate = f_hidden(gates[:, 2 * n_hidden_out:] + outpeep)

            s_t = inpt * ingate + s_tm1 * forgetgate
            h_t = f_hidden(s_t) * outgate
            return [s_t, h_t]

        inpt_flat = inpt.reshape((
            n_time_steps * n_samples, n_inpt))
        hidden_flat = T.dot(inpt_flat, in_to_hidden)
        hidden = hidden_flat.reshape((n_time_steps, n_samples, n_hidden_in))
        hidden += hidden_bias.dimshuffle('x', 'x', 0)

        (states, hidden_rec), _ = theano.scan(
            lstm_step,
            sequences=hidden,
            outputs_info=[T.zeros_like(hidden[0, :, 0:n_hidden_out]),
                          T.zeros_like(hidden[0, :, 0:n_hidden_out])
                          ])

        hidden_rec_flat = hidden_rec.reshape(
            (n_time_steps * n_samples, n_hidden_out))

        output_flat = T.dot(hidden_rec_flat, hidden_to_out)
        output_in = output_flat.reshape((n_time_steps, n_samples, n_output))
        output_in += out_bias.dimshuffle('x', 'x', 0)

        if pooling is None:
            pass
        elif pooling == 'mean':
            output_in = T.mean(output_in, axis=0)
        elif pooling == 'sum':
            output_in = T.sum(output_in, axis=0)
        elif pooling == 'prod':
            output_in = T.prod(output_in, axis=0)
        elif pooling == 'min':
            output_in = T.min(output_in, axis=0)
        elif pooling == 'max':
            output_in = T.max(output_in, axis=0)
        else:
            raise ValueError('unknown pooling operator %s' % pooling)

        output = f_output(output_in)

        loss = f_loss(output).sum(axis=2).mean()

        return {'inpt': inpt,
                'states': states,
                'hidden': hidden_rec,
                'output-in': output_in,
                'output': output,
                'loss': loss}
