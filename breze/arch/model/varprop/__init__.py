"""This package implements variance propagating networks.

If we really want to talk about neural networks in a probabilistic way, the
right way to do it is to treat every number in the network as a Dirac
distributed value.

There have been numerous attempts to model the adaptable parameters of networks
as random variables, leading to so called "Bayesian Neural Networks".

In some applications, it makes sense to treat the activations as random
variables. This can be done very efficiently and with a very good approximation
for the mean and the variance of random variables.

The algorithm for this has initially been described in [FD]_ and been described
in the context of RNNs in [FD-RNN]_.

References
----------

.. [FD] Wang, Sida, and Christopher Manning. "Fast dropout training."
        Proceedings of the 30th International Conference on Machine Learning
        (ICML-13). 2013.

.. [FD-RNN] Bayer, Justin, et al. "On Fast Dropout and its Applicability to
            Recurrent Networks." arXiv preprint arXiv:1311.0701 (2013).
"""

from mlp import VariancePropagationNetwork, FastDropoutNetwork
