# *- coding: utf-8 -*-


import theano.tensor as T

from ...util import lookup
from ...component.varprop import transfer, loss as loss_
from ...model.neural import MultiLayerPerceptron
from ...model.sequential import rnn


def dummy_complexity_loss(*args, **kwargs):
    return 0


def mean_var_forward(in_mean, in_var,
                     weights_mean, weights_var,
                     bias_mean, bias_var, f):
    weights_var = abs(weights_var)
    bias_var = abs(bias_var)

    out_in_mean = T.dot(in_mean, weights_mean)
    out_in_var = (
        T.dot(in_var, weights_mean ** 2)
        + T.dot(in_mean ** 2, weights_var)
        + T.dot(in_var, weights_var)
    )

    out_mean, out_var = f(out_in_mean, out_in_var)

    return out_in_mean, out_in_var, out_mean, out_var


class AdaptiveWeightNoiseNetwork(MultiLayerPerceptron):

    def __init__(self, n_inpt, n_hiddens, n_output,
                 hidden_transfers, out_transfer,
                 loss, complexity_loss):
        self.complexity_loss = complexity_loss
        super(AdaptiveWeightNoiseNetwork, self).__init__(
            n_inpt, n_hiddens, n_output,
            hidden_transfers, out_transfer,
            loss)

    @staticmethod
    def get_parameter_spec(n_inpt, n_hiddens, n_output):
        spec = {
            'in_to_hidden_mean': (n_inpt, n_hiddens[0]),
            'in_to_hidden_var': (n_inpt, n_hiddens[0]),

            'hidden_to_out_mean': (n_hiddens[-1], n_output),
            'hidden_to_out_var': (n_hiddens[-1], n_output),

            'hidden_bias_mean_0': n_hiddens[0],
            'hidden_bias_var_0': n_hiddens[0],
            'out_bias_mean': n_output,
            'out_bias_var': n_output,

            'prior_mean': 1,
            'prior_scale': 1,
        }

        zipped = zip(n_hiddens[:-1], n_hiddens[1:])
        spec['hidden_bias_0'] = n_hiddens[0]
        for i, (inlayer, outlayer) in enumerate(zipped):
            spec['hidden_to_hidden_mean_%i' % i] = (inlayer, outlayer)
            spec['hidden_to_hidden_var_%i' % i] = (inlayer, outlayer)
            spec['hidden_bias_mean_%i' % (i + 1)] = outlayer
            spec['hidden_bias_var_%i' % (i + 1)] = outlayer

        return spec

    def init_exprs(self):
        P = self.parameters   # Shortcut.
        hidden_to_hiddens_mean = [
            getattr(P, 'hidden_to_hidden_mean_%i' % i)
            for i in range(len(self.n_hiddens) - 1)]
        hidden_to_hiddens_var = [
            getattr(P, 'hidden_to_hidden_var_%i' % i)
            for i in range(len(self.n_hiddens) - 1)]
        hidden_biases_mean = [
            getattr(P, 'hidden_bias_mean_%i' % i)
            for i in range(len(self.n_hiddens))]
        hidden_biases_var = [
            getattr(P, 'hidden_bias_var_%i' % i)
            for i in range(len(self.n_hiddens))]

        inpt_mean = T.matrix('inpt_mean')
        inpt_var = T.zeros_like(inpt_mean) + self.inpt_var

        self.exprs = self.make_exprs(
            inpt_mean, inpt_var, T.matrix('target'),
            P.in_to_hidden_mean, P.in_to_hidden_var,
            hidden_to_hiddens_mean, hidden_to_hiddens_var,
            P.hidden_to_out_mean, P.hidden_to_out_var,
            hidden_biases_mean, hidden_biases_var,
            P.out_bias_mean, P.out_bias_var,
            self.hidden_transfers, self.out_transfer,
            self.loss, self.complexity_loss,
        )
        self.exprs['inpt'] = inpt_mean

    @staticmethod
    def make_exprs(inpt_mean, inpt_var, target,
                   in_to_hidden_mean, in_to_hidden_var,
                   hidden_to_hiddens_mean, hidden_to_hiddens_var,
                   hidden_to_out_mean, hidden_to_out_var,
                   hidden_biases_mean, hidden_biases_var,
                   out_bias_mean, out_bias_var,
                   hidden_transfers, out_transfer,
                   loss, complexity_loss):

        exprs = {}

        f_hidden = lookup(hidden_transfers[0], transfer)
        hidden = mean_var_forward(
            inpt_mean, inpt_var,
            in_to_hidden_mean, in_to_hidden_var,
            hidden_biases_mean[0], hidden_biases_var[0],
            f_hidden)

        (hidden_in_mean, hidden_in_var, hidden_mean, hidden_var) = hidden

        exprs['hidden_in_mean_0'] = hidden_in_mean
        exprs['hidden_in_var_0'] = hidden_in_var
        exprs['hidden_mean_0'] = hidden_mean
        exprs['hidden_var_0'] = hidden_var

        zipped = zip(hidden_to_hiddens_mean,
                     hidden_to_hiddens_var,
                     hidden_biases_mean[1:],
                     hidden_biases_var[1:],
                     hidden_transfers[1:])
        for i, (wm, wv, bm, bv, t) in enumerate(zipped):
            hidden_mean_m1, hidden_var_m1 = hidden_mean, hidden_var
            f = lookup(t, transfer)
            hidden = mean_var_forward(
                hidden_mean_m1, hidden_var_m1, wm, wv, bm, bv, f)
            (hidden_in_mean, hidden_in_var, hidden_mean, hidden_var) = hidden

            exprs['hidden_in_mean_%i' % (i + 1)] = hidden_in_mean
            exprs['hidden_in_var_%i' % (i + 1)] = hidden_in_var
            exprs['hidden_mean_%i' % (i + 1)] = hidden_mean
            exprs['hidden_var_%i' % (i + 1)] = hidden_var

        f_output = lookup(out_transfer, transfer)
        output = mean_var_forward(
            hidden_mean, hidden_var, hidden_to_out_mean, hidden_to_out_var,
            out_bias_mean, out_bias_var, f_output)

        (output_in_mean, output_in_var, output_mean, output_var) = output
        output = T.concatenate([output_mean, output_var], axis=1)

        f_loss = lookup(loss, loss_)

        loss_rowwise = f_loss(target, output).sum(axis=1)

        all_pars = [
            (in_to_hidden_mean, in_to_hidden_var),
            (hidden_to_out_mean, hidden_to_out_var),
            (out_bias_mean, out_bias_var)]
        all_pars += zip(hidden_biases_mean, hidden_biases_var)
        all_pars += zip(hidden_to_hiddens_mean, hidden_to_hiddens_var)

        loss = loss_rowwise.mean() + dummy_complexity_loss(all_pars)

        exprs.update({
            'inpt_mean': inpt_mean,
            'inpt_var': inpt_var,
            'target': target,
            'output_in_mean': output_in_mean,
            'output_in_var': output_in_var,
            'output_mean': output_in_mean,
            'output_var': output_in_var,
            'output': output,
            'loss_rowwise': loss_rowwise,
            'loss': loss
        })

        return exprs
