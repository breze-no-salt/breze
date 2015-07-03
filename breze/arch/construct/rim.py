# -*- coding: utf-8 -*-

"""Module that offers functionality for RIM.

The relevant reference is

    Gomes, Ryan, Andreas Krause, and Pietro Perona.
    "Discriminative Clustering by Regularized Information Maximization."
    NIPS. 2010.
"""


from ..component import misc
from breze.arch.construct.base import Layer


class RimLoss(Layer):

    def __init__(self, posterior, pars_to_penalize, c_rim,
                 comp_dim=1, imp_weight=None, declare=None, name=None):
        """Create a Loss object representing the Regularized Information
        Maximization (RIM) loss of a set of categorical distributions.

        Parameters
        ----------

        posterior : Theano variable
            Array of the shape ``(n, d)``. The array describes ``n`` catgorical
            distributions over ``d`` categories. Each row has thus to sum up to
            1 and each entry has to be non-negative.

        pars_to_penalize : list of Theano variables
            Each of the items is a Theano variable that is being penalized with
            its squared L2 norm.

        c_rim : float
            Weight of the L2 penalties.
        """
        self.posterior = posterior
        self.pars_to_penalize = pars_to_penalize
        self.c_rim = c_rim
        self.comp_dim = comp_dim

        super(RimLoss, self).__init__(declare, name)

    def _forward(self):
        marginal = self.posterior.mean(axis=0)
        cond_entropy = misc.cat_entropy(self.posterior).mean()
        entropy = misc.cat_entropy(marginal.dimshuffle('x', 0)).sum()

        nmi = cond_entropy - entropy

        n_samples = self.posterior.shape[0]
        penalties = [(i ** 2).sum() / n_samples for i in self.pars_to_penalize]
        penalty = sum(penalties)

        self.total = nmi + self.c_rim * penalty
