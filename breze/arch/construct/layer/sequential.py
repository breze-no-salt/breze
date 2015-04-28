# -*- coding: utf-8 -*-


from breze.arch.component import transfer as _transfer
from breze.arch.construct.base import Layer
from breze.arch.model.rnn.rnn import recurrent_layer
from breze.arch.model.rnn.pooling import pooling_layer
from breze.arch.util import get_named_variables, lookup


class SequentialToStatic(Layer):

    def forward(self, *inpts):
        self.n_time_steps = inpts[0].shape[0]
        outputs = [i.reshape((-1, i.shape[2])) for i in inpts]
        self.exprs = get_named_variables(locals())
        self.output = outputs

    def inverse(self, *args):
        return [i.reshape((self.n_time_steps, -1, i.shape[1])) for i in args]


class Recurrent(Layer):

    def __init__(self, n_inpt, transfer='identity', name=None):
        self.n_inpt = n_inpt
        self.transfer = transfer
        super(Recurrent, self).__init__(name)

    def spec(self):
        return {
            'weights': (self.n_inpt, self.n_inpt),
            'initial': (self.n_inpt,),
        }

    def forward(self, inpt):
        super(Recurrent, self).forward(inpt)

        weights = self.parameterized('weights', (self.n_inpt, self.n_inpt))
        initial = self.parameterized('initial', (self.n_inpt,))

        f_transfer = lookup(self.transfer, _transfer)
        presynaptic, output = recurrent_layer(
            inpt, weights, f_transfer, initial)

        E = self.exprs = get_named_variables(locals())
        self.output = [output]


class Pooling(Layer):

    def __init__(self, typ='mean', name=None):
        self.typ = typ
        super(Pooling, self).__init__(name)

    def forward(self, inpt):
        output = pooling_layer(inpt, self.typ)
        self.exprs = get_named_variables(locals())
        self.output = [output]
