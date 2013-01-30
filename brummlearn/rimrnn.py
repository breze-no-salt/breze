import theano.tensor as T

from .rnn import UnsupervisedRnn
from breze.component.misc import discrete_entropy

def dummy_loss(X):
    return X.sum()


class RimRnn(UnsupervisedRnn):

    def __init__(self, n_inpt, n_hidden, n_output,
                 c_l2,
                 hidden_transfer='tanh',
                 pooling='mean',
                 leaky_coeffs=None,
                 optimizer='rprop',
                 max_iter=1000,
                 verbose=False):
        self.c_l2 = c_l2
        super(RimRnn, self).__init__(
            n_inpt, n_hidden, n_output, hidden_transfer, 'softmax',
            dummy_loss, pooling, leaky_coeffs, optimizer, None,
            max_iter, verbose)

    def init_exprs(self):
        super(RimRnn, self).init_exprs()

        output = self.exprs['output']
        weights = self.parameters.hidden_to_out

        marginal = output.mean(axis=0)
        cond_entropy = discrete_entropy(output, axis=1).mean()
        entropy = discrete_entropy(marginal)

        # negative mutual information -> we are minimizing
        neg_mi = cond_entropy - entropy
        l2 = (weights**2).sum() / output.shape[0]

        self.exprs.update({
            'neg_mi': neg_mi,
            'l2': l2,
            'loss': neg_mi + self.c_l2 * l2,
        })
