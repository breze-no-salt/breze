# -*- coding: utf-8 -*-


from breze.arch.component import transfer as _transfer
from breze.arch.construct.base import Layer
from breze.arch.model.rnn.rnn import recurrent_layer
from breze.arch.model.rnn.pooling import pooling_layer
from breze.arch.util import lookup


class Recurrent(Layer):

    def __init__(self, inpt, n_inpt, transfer='identity', declare=None,
                 name=None):
        self.inpt = inpt
        self.n_inpt = n_inpt
        self.transfer = transfer
        super(Recurrent, self).__init__(declare=declare, name=name)

    def _forward(self):

        weights = self.declare((self.n_inpt, self.n_inpt))
        initial = self.declare((self.n_inpt,))

        f = lookup(self.transfer, _transfer)
        self.output_in, self.output = recurrent_layer(
            self.inpt, weights, f, initial)


class Pooling(Layer):

    def __init__(self, inpt, typ='mean', declare=None, name=None):
        self.inpt = inpt
        self.typ = typ
        super(Pooling, self).__init__(declare=None, name=name)

    def _forward(self):
        self.output = pooling_layer(self.inpt, self.typ)
