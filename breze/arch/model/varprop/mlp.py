# -*- coding: utf-8 -*-


import theano.tensor as T

from ...util import lookup
from ...component.varprop import transfer, loss as loss_
from ...model.neural import MultiLayerPerceptron
from ...model.sequential import rnn


def mean_var_forward(in_mean, in_var, weights, bias, variance_bias_sqrt,
                     f, p_dropout):
    p_keep = 1 - p_dropout
    out_in_mean = T.dot(in_mean, weights) * p_keep
    out_in_mean += bias

    dropout_var = p_dropout * (1 - p_dropout)

    element_var = (dropout_var * in_var
                   + in_mean ** 2 * dropout_var
                   + in_var * p_keep ** 2)

    out_in_var = T.dot(element_var, weights ** 2)

    out_in_var *= variance_bias_sqrt ** 2
    out_mean, out_var = f(out_in_mean, out_in_var)
    return out_in_mean, out_in_var, out_mean, out_var


def int_mean_var_forward(in_mean, in_var, weights, bias, variance_bias_sqrt,
                         f, p_dropout):
    p_keep = 1 - p_dropout
    dropout_var = p_dropout * (1 - p_dropout)

    out_in_mean = p_keep * weights[in_mean] + bias
    out_in_var = dropout_var * weights[in_mean] ** 2# + p_keep * in_var[in_mean]

    #out_in_var *= variance_bias_sqrt ** 2
    out_mean, out_var = f(out_in_mean, out_in_var)
    return out_in_mean, out_in_var, out_mean, out_var



class VariancePropagationNetwork(MultiLayerPerceptron):

    def init_exprs(self):
        hidden_to_hiddens = [getattr(self.parameters, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(self.parameters, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        hidden_var_biases = [getattr(self.parameters, 'hidden_var_bias_%i' % i)
                             for i in range(len(self.n_hiddens))]
        inpt = T.matrix('inpt')
        inpt_mean = inpt[:, :inpt.shape[1] // 2]
        inpt_var = inpt[:, inpt.shape[1] // 2:]
        self.exprs = self.make_exprs(
            inpt_mean, inpt_var, T.matrix('target'),
            self.parameters.in_to_hidden,
            hidden_to_hiddens,
            self.parameters.hidden_to_out,
            hidden_biases,
            hidden_var_biases,
            self.parameters.out_bias,
            self.parameters.out_var_bias,
            self.hidden_transfers, self.out_transfer, self.loss,
            # Workaround for no dropout: very small dropout.
            1e-32, 1e-32)

        self.exprs['inpt'] = inpt

    @staticmethod
    def get_parameter_spec(n_inpt, n_hiddens, n_output):
        spec = MultiLayerPerceptron.get_parameter_spec(
            n_inpt, n_hiddens, n_output)

        zipped = zip(n_hiddens[:-1], n_hiddens[1:])
        spec['hidden_var_bias_0'] = n_hiddens[0]
        for i, (inlayer, outlayer) in enumerate(zipped):
            spec['hidden_var_bias_%i' % (i + 1)] = outlayer

        spec['out_var_bias'] = n_output

        return spec

    @staticmethod
    def make_exprs(inpt_mean, inpt_var, target, in_to_hidden,
                   hidden_to_hiddens,
                   hidden_to_out,
                   hidden_biases,
                   hidden_var_biases,
                   out_bias,
                   out_var_bias,
                   hidden_transfers, output_transfer, loss,
                   p_dropout_inpt,
                   p_dropout_hidden):
        exprs = {}

        f_hidden = lookup(hidden_transfers[0], transfer)
        hidden = mean_var_forward(inpt_mean, inpt_var, in_to_hidden,
                                  hidden_biases[0], hidden_var_biases[0],
                                  f_hidden, p_dropout_inpt)
        (hidden_in_mean, hidden_in_var, hidden_mean, hidden_var) = hidden

        exprs['hidden_in_mean_0'] = hidden_in_mean
        exprs['hidden_in_var_0'] = hidden_in_var
        exprs['hidden_mean_0'] = hidden_mean
        exprs['hidden_var_0'] = hidden_var

        zipped = zip(hidden_to_hiddens,
                     hidden_biases[1:],
                     hidden_var_biases[1:],
                     hidden_transfers[1:])
        for i, (w, b, bv, t) in enumerate(zipped):
            hidden_mean_m1, hidden_var_m1 = hidden_mean, hidden_var
            f = lookup(t, transfer)
            hidden = mean_var_forward(hidden_mean_m1, hidden_var_m1, w, b, bv,
                                      f, p_dropout_hidden)
            (hidden_in_mean, hidden_in_var, hidden_mean, hidden_var) = hidden

            exprs['hidden_in_mean_%i' % (i + 1)] = hidden_in_mean
            exprs['hidden_in_var_%i' % (i + 1)] = hidden_in_var
            exprs['hidden_mean_%i' % (i + 1)] = hidden_mean
            exprs['hidden_var_%i' % (i + 1)] = hidden_var

        f_output = lookup(output_transfer, transfer)
        output = mean_var_forward(hidden_mean, hidden_var, hidden_to_out,
                                  out_bias, out_var_bias,
                                  f_output, p_dropout_hidden)
        (output_in_mean, output_in_var, output_mean, output_var) = output
        output = T.concatenate([output_mean, output_var], axis=1)

        f_loss = lookup(loss, loss_)

        loss_rowwise = f_loss(target, output).sum(axis=1)
        loss = loss_rowwise.mean()

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


class SupervisedRecurrentNetwork(rnn.BaseRecurrentNetwork,
                                 rnn.SimpleRnnComponent):

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


class FastDropoutNetwork(VariancePropagationNetwork):

    inpt_var = 0
    p_dropout_inpt = .2
    p_dropout_hidden = .5

    # This method overwrites the VariancePropagationNetwork's get_parameter_spec
    # with the MultiLayerPerceptron's one. This is not very nice OOP, but we
    # need to get rid of the hidden variance bias.
    @staticmethod
    def get_parameter_spec(n_inpt, n_hiddens, n_output):
        spec = MultiLayerPerceptron.get_parameter_spec(
            n_inpt, n_hiddens, n_output)

        return spec

    def init_exprs(self):
        hidden_to_hiddens = [getattr(self.parameters, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(self.parameters, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        inpt_mean = T.matrix('inpt_mean')
        inpt_var = T.zeros_like(inpt_mean) + self.inpt_var

        self.exprs = self.make_exprs(
            inpt_mean, inpt_var, T.matrix('target'),
            self.parameters.in_to_hidden,
            hidden_to_hiddens,
            self.parameters.hidden_to_out,
            hidden_biases,
            [1 for _ in hidden_biases],
            self.parameters.out_bias,
            1,
            self.hidden_transfers, self.out_transfer, self.loss,
            self.p_dropout_inpt, self.p_dropout_hidden)

        self.exprs['inpt'] = inpt_mean
