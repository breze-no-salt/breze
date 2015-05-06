# -*- coding: utf-8 -*-

import theano.tensor as T

from breze.arch.component import transfer as _transfer
from breze.arch.construct.base import Layer


class VariationalAutoEncoder(Layer):

    def __init__(self, inpt, n_inpt, n_latent, n_output, assumptions, recog_class, gen_class, condition=None, declare=None, name=None):
        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_latent = n_latent
        self.n_output = n_output
        self.assumptions = assumptions
        self.recog_class = recog_class
        self.gen_class = gen_class
        self.condition = condition
        self.transfer = _transfer
        super(VariationalAutoEncoder, self).__init__(declare=declare, name=name)

    def _forward(self):
        rng = T.shared_randomstreams.RandomStreams()

        # recognition model + sampling
        self.recog = self.recog_class(self.inpt,self.declare)
        self.latent = self.recog.output
        self.sample = self.assumptions.sample_latents(self.latent,rng)

        if self.condition is None:
            gen_inpt = self.sample 
        else:
            gen_inpt = T.concatenate([self.sample, self.condition],
                                 axis=latent.ndim - 1)

        #generative model
        self.gen = self.gen_class(gen_inpt,self.declare)
        self.output = self.gen.output