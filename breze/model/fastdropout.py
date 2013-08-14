# -*- coding: utf-8 -*-


import theano.tensor as T

from ..util import lookup
from ..component import meanvartransfer, loss as loss_
from ..model.neural import MultiLayerPerceptron


def mean_var_forward(in_mean, in_var, weights, bias, variance_bias, transfer,
                     p_dropout):
    out_in_mean = T.dot(in_mean, weights) * p_dropout
    out_in_mean += bias

    dropout_var = p_dropout * (1 - p_dropout)
    out_in_var = (T.dot(in_mean ** 2, weights ** 2) * dropout_var
                  + T.dot(in_var, weights ** 2) * p_dropout)
    out_in_var *= T.exp(variance_bias)
    out_mean, out_var = transfer(out_in_mean, out_in_var)
    return out_in_mean, out_in_var, out_mean, out_var


class MeanVarianceNetwork(MultiLayerPerceptron):

    def init_exprs(self):
        hidden_to_hiddens = [getattr(self.parameters, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(self.parameters, 'hidden_bias_%i' % i)
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
            self.parameters.out_bias,
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

        f_hidden = lookup(hidden_transfers[0], meanvartransfer)
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
            f = lookup(t, meanvartransfer)
            hidden = mean_var_forward(hidden_mean_m1, hidden_var_m1, w, b, bv,
                                      f, p_dropout_hidden)
            (hidden_in_mean, hidden_in_var, hidden_mean, hidden_var) = hidden

            exprs['hidden_in_mean_%i' % (i + 1)] = hidden_in_mean
            exprs['hidden_in_var_%i' % (i + 1)] = hidden_in_var
            exprs['hidden_mean_%i' % (i + 1)] = hidden_mean
            exprs['hidden_var_%i' % (i + 1)] = hidden_var

        f_output = lookup(output_transfer, meanvartransfer)
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


class FastDropoutNetwork(MeanVarianceNetwork):

    inpt_var = 0
    var_bias_offset = 0.0

    def init_exprs(self):
        hidden_to_hiddens = [getattr(self.parameters, 'hidden_to_hidden_%i' % i)
                             for i in range(len(self.n_hiddens) - 1)]
        hidden_biases = [getattr(self.parameters, 'hidden_bias_%i' % i)
                         for i in range(len(self.n_hiddens))]
        hidden_var_biases = [getattr(self.parameters, 'hidden_var_bias_%i' % i)
                             for i in range(len(self.n_hiddens))]
        inpt_mean = T.matrix('inpt_mean')
        inpt_var = T.zeros_like(inpt_mean) + self.inpt_var

        # Clamp the variance biases to a minimal value.
        hidden_var_biases = [T.log(T.exp(i) + self.var_bias_offset)
                             for i in hidden_var_biases]
        out_var_bias = T.log(
            T.exp(self.parameters.out_var_bias) + self.var_bias_offset)

        self.exprs = self.make_exprs(
            inpt_mean, inpt_var, T.matrix('target'),
            self.parameters.in_to_hidden,
            hidden_to_hiddens,
            self.parameters.hidden_to_out,
            hidden_biases,
            hidden_var_biases,
            self.parameters.out_bias,
            out_var_bias,
            self.hidden_transfers, self.out_transfer, self.loss,
            self.p_dropout_inpt, self.p_dropout_hidden)

        self.exprs['inpt'] = inpt_mean
