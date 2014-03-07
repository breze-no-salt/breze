# -*- coding: utf-8 -*-

"""ICA with reconstruction cost.

As introduced in [rica]_.

References
----------

.. [rica] ICA with Reconstruction Cost for Efficient Overcomplete Feature
   Learning. Quoc V. Le, Alex Karpenko, Jiquan Ngiam and Andrew Y. Ng. In NIPS
   2011.
"""

from climin import mathadapt as ma
import numpy as np
import theano

from breze.arch.model.feature import rica
from breze.learn.base import (
    UnsupervisedBrezeWrapperBase, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)
from breze.arch.util import ParameterSet, Model, lookup, get_named_variables
import autoencoder


class Rica(autoencoder.AutoEncoder):
    """Class implementing ICA with reconstruction cost.

    All attributes of ``AutoEncoder`` objects apply as well.

    Attributes
    ----------

    code_transfer : string or function
        Transfer function to use for the features before the loss. Can be a
        string referring any function found in ``breze.component.transfer`` or
        a function that given an (n, d) array returns an (n, d) array as
        theano expressions.

    c_ica : float, optional, [default: 0.5]
        Weight of the ICA cost, cost of reconstruction is 1.

    fit_project_weights :  boolean
        If set to True, the length of an weight vector coming into a code unit
        will be set to 1 after each training iteration.
    """

    def __init__(self, n_inpt, n_hidden, hidden_transfer='identity',
                 code_transfer='softabs', out_transfer='identity',
                 loss='squared', c_ica=0.5,
                 fit_project_weights=True,
                 tied_weights=True,
                 batch_size=None,
                 optimizer='lbfgs',
                 max_iter=1000, verbose=False):
        """Create a Rica object.

        Parameters
        ----------

        All parameters from the ``AutoEncoder`` class apply as well.

        code_transfer : string or function
            Transfer function to use for the features before the loss. Can be a
            string referring any function found in ``breze.component.transfer``
            or a function that given an (n, d) array returns an (n, d) array as
            theano expressions.

        c_ica : float, optional, [default: 0.5]
            Weight of the ICA cost, cost of linear reconstruction is 1.

        fit_project_weights : boolean
            If set to True, the length of an weight vector coming into a code
            unit will be set to 1 after each training iteration.
        """
        self.code_transfer = code_transfer
        self.c_ica = c_ica
        self.fit_project_weights = fit_project_weights
        super(Rica, self).__init__(
            n_inpt, [n_hidden], [hidden_transfer],
            out_transfer, loss, tied_weights=tied_weights,
            batch_size=batch_size,
            code_idx=None)

    def _init_exprs(self):
        super(Rica, self)._init_exprs()
        self.exprs.update(rica.loss(self.exprs['layer-0-output'],
                                    self.code_transfer))

        self.exprs['loss_coord_wise'] += self.exprs['ica_loss_coord_wise']
        self.exprs['loss_sample_wise'] += self.exprs['ica_loss_sample_wise']
        self.exprs['loss'] += self.exprs['ica_loss']

    def project_weights(self):
        w = self.parameters['in_to_hidden']
        w /= ma.sqrt((w ** 2).sum(axis=0) + 1e-4)[np.newaxis, :]

    def iter_fit(self, X):
        for info in super(Rica, self).iter_fit(X):
            if self.fit_project_weights:
                self.project_weights()
            yield info
