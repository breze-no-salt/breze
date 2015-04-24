# -*- coding: utf-8 -*-


from breze.arch.construct.base import Layer
from breze.arch.util import get_named_variables


class SequentialToStatic(Layer):

    def forward(self, *inpts):
        self.n_time_steps = inpts[0].shape[0]
        outputs = [i.reshape((-1, i.shape[2])) for i in inpts]
        self.exprs = get_named_variables(locals())
        self.output = outputs

    def inverse(self, *args):
        return [i.reshape((self.n_time_steps, -1, i.shape[1])) for i in args]
