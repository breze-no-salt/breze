# -*- coding: utf-8 -*-


from breze.arch.component import transfer as _transfer
from breze.arch.construct.base import Layer
from breze.arch.model.rnn.rnn import recurrent_layer, recurrent_layer_stateful
from breze.arch.model.rnn.pooling import pooling_layer
from breze.arch.util import lookup


class Recurrent(Layer):
    """Recurrent class.

    Represents a recurrent layer as found in neural networks.
    """

    def __init__(self, inpt, n_inpt, transfer='identity', declare=None,
                 name=None):
        self.inpt = inpt
        self.n_inpt = n_inpt
        self.transfer = transfer
        super(Recurrent, self).__init__(declare=declare, name=name)

    def _forward(self):
        f = lookup(self.transfer, _transfer)
        n, m = self.n_inpt, self.n_inpt

        # If a transfer function has differing dimensionalities in its domain
        # and co-domain, it can be specified by its ``in_size`` and ``out_size``
        # attributes. Defaulting to not to.
        n *= getattr(f, 'in_size', 1)
        m *= getattr(f, 'out_size', 1)

        self.weights = self.declare((m, n))
        self.initial = self.declare((m,))

        if getattr(f, 'stateful', False):
            self.state, self.output_in, self.output = recurrent_layer_stateful(
                self.inpt, self.weights, f, self.initial)
        else:
            self.output_in, self.output = recurrent_layer(
                self.inpt, self.weights, f, self.initial)


class Pooling(Layer):

    def __init__(self, inpt, typ='mean', declare=None, name=None):
        self.inpt = inpt
        self.typ = typ
        super(Pooling, self).__init__(declare=None, name=name)

    def _forward(self):
        self.output = pooling_layer(self.inpt, self.typ)
