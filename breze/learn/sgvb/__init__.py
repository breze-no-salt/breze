# -*- coding: utf-8 -*-

"""Module for learning models with stochastic gradient variational Bayes (SGVB).

The method has been introduced in parallel by [SGVB]_ and [DLGM]_. The general
idea is to optimize a variational upper bound on the negative log-likelihood of
the data with stochastic gradient descent. The user is urged to review these
papers before using the models.


Methodological review
---------------------

We will give a short review for notation here. We consider the problem of
estimating a model for the data density :math:`p(x)`, where we assume it to be
driven by a set of latent variables :math:`z`. The data negative log likelihood
can then be bounded from above

.. math::
   -\\log p(x) \\le {\\text{KL}}(q(z|x)|p(z)) - \mathbb{E}_{z \\sim q}[\\log p(x|z)].

We will refer to :math:`q(z)` as the recognition model or approxiamte posterior.
:math:`p(x|z)` is the generating model. :math:`p(z)` is the prior over the
latent variables.
The first term is the Kullback-Leibler divergence between the approximate
posterior and the prior.
The second term is the expected negative log-likelihood of the data given the
recognition model.

Training of the model is performed by stochastic gradient descent. In practice,
we will select a mini batch of the data for which we can obtain :math:`q(z|x)`.
This can directly be used to calculate the first term. By sampling from that
model and putting it through the generating model we can approximate the second
term.


References
----------

.. [SGVB] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
.. [DLGM] Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic Back-propagation and Variational Inference in Deep Latent Gaussian Models." arXiv preprint arXiv:1401.4082 (2014).
"""


from mlp import (
    VariationalAutoEncoder, MlpGaussLatentVAEMixin, MlpGaussVisibleVAEMixin,
    MlpGaussConstVarVisibleVAEMixin,
    MlpBernoulliVisibleVAEMixin,
    FastDropoutVariationalAutoEncoder,
    FastDropoutMlpGaussLatentVAEMixin, FastDropoutMlpGaussVisibleVAEMixin,
    FastDropoutMlpBernoulliVisibleVAEMixin)

from storn import (
    StochasticRnn, BernoulliVisibleStornMixin, GaussLatentStornMixin,
    GaussVisibleStornMixin, ConstVarGaussVisibleStornMixin,
    GaussLatentBiStornMixin)
