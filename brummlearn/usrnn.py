import theano.tensor as T

from .rnn import UnsupervisedRnn
from breze.model.sequential.rnn import weighted_pooling
from breze.component.misc import discrete_entropy
from breze.component import norm


def dummy_loss(X):
    return X.sum()


class RimRnn(UnsupervisedRnn):

    def __init__(self, n_inpt, n_hidden, n_output,
                 c_l2,
                 hidden_transfer='tanh',
                 pooling='mean',
                 leaky_coeffs=None,
                 optimizer='rprop',
                 batch_size=None,
                 max_iter=1000,
                 verbose=False):
        self.c_l2 = c_l2
        super(RimRnn, self).__init__(
            n_inpt, n_hidden, n_output, hidden_transfer, 'softmax',
            dummy_loss, pooling, leaky_coeffs, optimizer, batch_size,
            max_iter, verbose)

    def init_exprs(self):
        super(RimRnn, self).init_exprs()

        output = self.exprs['output']
        weights = self.parameters.hidden_to_out

        marginal = output.mean(axis=0)
        entropy = discrete_entropy(marginal)

        cond_entropy = discrete_entropy(output, axis=1).mean()

        # negative mutual information -> we are minimizing
        neg_mi = cond_entropy - entropy
        l2 = (weights**2).sum()

        self.exprs.update({
            'neg_mi': neg_mi,
            'l2': l2,
            'loss': neg_mi + self.c_l2 * l2,
        })


class SfRnn(UnsupervisedRnn):

    def __init__(self, n_inpt, n_hidden, n_output,
                 hidden_transfer='tanh',
                 out_transfer='softabs',
                 pooling='mean',
                 leaky_coeffs=None,
                 optimizer='rprop',
                 batch_size=None,
                 max_iter=1000,
                 verbose=False):
        super(SfRnn, self).__init__(
            n_inpt, n_hidden, n_output, hidden_transfer, out_transfer,
            dummy_loss, pooling, leaky_coeffs, optimizer, batch_size,
            max_iter, verbose)

    def init_exprs(self):
        super(SfRnn, self).init_exprs()

        output = self.exprs['output']

        col_normalized = T.sqrt(
            norm.normalize(output, lambda x: x**2, axis=0) + 1E-8)
        row_normalized = T.sqrt(
            norm.normalize(col_normalized, lambda x: x**2, axis=1) + 1E-8)

        loss_rowwise = row_normalized.sum(axis=1)
        loss = loss_rowwise.mean()

        if self.pooling == 'stochastic':
            representation = weighted_pooling(self.exprs['unpooled'])
        else:
            representation = output

        self.exprs.update({
            'col_normalized': col_normalized,
            'row_normalized': row_normalized,
            'loss_rowwise': loss_rowwise,
            'representation': representation,
            'loss': loss})
