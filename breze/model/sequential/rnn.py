# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T

from ...util import ParameterSet, Model, lookup
from ...component import transfer, distance


class RecurrentNetwork(Model):

    def __init__(self, n_inpt, n_hidden, n_output, 
                 hidden_transfer, out_transfer, loss, pooling=None):
        self.n_inpt = n_inpt
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hidden_transfer = hidden_transfer
        self.out_transfer = out_transfer
        self.loss = loss
        self.pooling = pooling
        super(RecurrentNetwork, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(
            self.n_inpt, self.n_hidden, self.n_output)
        self.parameters = ParameterSet(**parspec)

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
            self.hidden_transfer, self.out_transfer, self.loss,
            self.pooling)

    @staticmethod
    def get_parameter_spec(n_inpt, n_hidden, n_output):
        return {
            'in_to_hidden': (n_inpt, n_hidden),
            'hidden_to_hidden': (n_hidden, n_hidden),
            'hidden_to_out': (n_hidden, n_output),
            'hidden_bias': n_hidden,
            'out_bias': n_output,
        }

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

        loss = f_loss(output, target)

        return {'inpt': inpt,
                'target': target,
                'hidden-in': hidden,
                'hidden-in-rec': hidden_in_rec,
                'hidden': hidden_rec,
                'output-in': output_in,
                'output': output,
                'loss': loss}
