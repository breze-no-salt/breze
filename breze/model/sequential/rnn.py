# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance


class BaseRecurrentNetwork(Model):

    def __init__(self, n_inpt, n_hidden, n_output, 
                 hidden_transfer, out_transfer='identity', loss='squared',
                 pooling=None):
        self.n_inpt = n_inpt
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hidden_transfer = hidden_transfer
        self.out_transfer = out_transfer
        self.loss = loss
        self.pooling = pooling
        super(BaseRecurrentNetwork, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(
            self.n_inpt, self.n_hidden, self.n_output)
        self.parameters = ParameterSet(**parspec)

    @staticmethod
    def get_parameter_spec(n_inpt, n_hidden, n_output):
        return {
            'in_to_hidden': (n_inpt, n_hidden),
            'hidden_to_hidden': (n_hidden, n_hidden),
            'hidden_to_out': (n_hidden, n_output),
            'hidden_bias': n_hidden,
            'out_bias': n_output,
        }


class SupervisedRecurrentNetwork(BaseRecurrentNetwork):

    def init_exprs(self):
        inpt = T.tensor3('inpt')
        if self.pooling is None:
            target = T.tensor3('target') 
        else:
            target = T.matrix('target')
        pars = self.parameters
        self.exprs = self.make_exprs(
            inpt, target,
            pars.in_to_hidden, pars.hidden_to_hidden, pars.hidden_to_out,
            pars.hidden_bias, pars.out_bias, 
            self.hidden_transfer, self.out_transfer, self.loss,
            self.pooling)

    @staticmethod
    def make_exprs(inpt, target, in_to_hidden, hidden_to_hidden, hidden_to_out,
                   hidden_bias, out_bias, hidden_transfer, out_transfer,
                   loss, pooling):

        f_hidden = lookup(hidden_transfer, transfer)
        f_output = lookup(out_transfer, transfer)
        f_loss = lookup(loss, distance)

        n_time_steps = inpt.shape[0]
        n_samples = inpt.shape[1]
        n_inpt = in_to_hidden.shape[0]
        n_hidden = in_to_hidden.shape[1]
        n_output = hidden_to_out.shape[1]

        inpt_flat = inpt.reshape((n_time_steps * n_samples, n_inpt))
        hidden_flat = T.dot(inpt_flat, in_to_hidden)
        hidden = hidden_flat.reshape((n_time_steps, n_samples, n_hidden))
        hidden += hidden_bias.dimshuffle('x', 'x', 0)

        def step(x, hi_tm1):
          h_tm1 = f_hidden(hi_tm1)
          hi = T.dot(h_tm1, hidden_to_hidden) + x
          return hi

        hidden_in_rec, _ = theano.scan(
          step,
          sequences=hidden,
          outputs_info=[T.zeros_like(hidden[0])])

        hidden_rec = f_hidden(hidden_in_rec)

        hidden_rec_flat = hidden_rec.reshape(
            (n_time_steps * n_samples, n_hidden))

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
            raise ValueError('unknown pooling operator %s' % self.pooling)

        output = f_output(output_in)

        loss = f_loss(target, output)

        return {'inpt': inpt,
                'target': target,
                'hidden-in': hidden,
                'hidden-in-rec': hidden_in_rec,
                'hidden': hidden_rec,
                'output-in': output_in,
                'output': output,
                'loss': loss}


class UnsupervisedRecurrentNetwork(BaseRecurrentNetwork):

    def init_exprs(self):
        inpt = T.tensor3('inpt')
        pars = self.parameters
        self.exprs = self.make_exprs(
            inpt,
            pars.in_to_hidden, pars.hidden_to_hidden, pars.hidden_to_out,
            pars.hidden_bias, pars.out_bias, 
            self.hidden_transfer, self.out_transfer, self.loss,
            self.pooling)

    @staticmethod
    def make_exprs(inpt, in_to_hidden, hidden_to_hidden, hidden_to_out,
                   hidden_bias, out_bias, hidden_transfer, out_transfer,
                   loss, pooling):

        f_hidden = lookup(hidden_transfer, transfer)
        f_output = lookup(out_transfer, transfer)
        f_loss = lookup(loss, distance)

        n_time_steps = inpt.shape[0]
        n_samples = inpt.shape[1]
        n_inpt = in_to_hidden.shape[0]
        n_hidden = in_to_hidden.shape[1]
        n_output = hidden_to_out.shape[1]

        inpt_flat = inpt.reshape((n_time_steps * n_samples, n_inpt))
        hidden_flat = T.dot(inpt_flat, in_to_hidden)
        hidden = hidden_flat.reshape((n_time_steps, n_samples, n_hidden))
        hidden += hidden_bias.dimshuffle('x', 'x', 0)

        def step(x, hi_tm1):
          h_tm1 = f_hidden(hi_tm1)
          hi = T.dot(h_tm1, hidden_to_hidden) + x
          return hi

        hidden_in_rec, _ = theano.scan(
          step,
          sequences=hidden,
          outputs_info=[T.zeros_like(hidden[0])])

        hidden_rec = f_hidden(hidden_in_rec)

        hidden_rec_flat = hidden_rec.reshape(
            (n_time_steps * n_samples, n_hidden))

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
        elif pooling == 'last':
            output_in = output_in[-1]
        else:
            raise ValueError('unknown pooling operator %s' % self.pooling)

        output = f_output(output_in)

        loss = f_loss(output)

        return {'inpt': inpt,
                'hidden-in': hidden,
                'hidden-in-rec': hidden_in_rec,
                'hidden': hidden_rec,
                'output-in': output_in,
                'output': output,
                'loss': loss}



class LstmRecurrentNetwork(SupervisedRecurrentNetwork):

    def __init__(self, n_inpt, n_hidden, n_output, 
                 hidden_transfer, out_transfer='identity', loss='squared',
                 pooling=None):
        super(LstmRecurrentNetwork, self).__init__(
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
        in_to_hidden, hidden_to_hidden, hidden_to_out, hidden_bias, out_bias,
        ingate_peephole, outgate_peephole, forgetgate_peephole,
        hidden_transfer, out_transfer, loss, pooling):

        f_hidden = lookup(hidden_transfer, transfer)
        f_output = lookup(out_transfer, transfer)
        f_loss = lookup(loss, distance)

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
            forgetgate = f_hidden(gates[:, n_hidden_out:2 * n_hidden_out] + forgetpeep)
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
            raise ValueError('unknown pooling operator %s' % self.pooling)

        output = f_output(output_in)

        loss = f_loss(target, output)

        return {'inpt': inpt,
                'target': target,
                'states': states,
                'hidden': hidden_rec,
                'output-in': output_in,
                'output': output,
                'loss': loss}


