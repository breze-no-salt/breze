# -*- coding: utf-8 -*-

import theano.tensor as T

from breze.arch.component import transfer as _transfer
from breze.arch.construct.base import Layer


class VariationalAutoEncoder(Layer):

    def __init__(self, inpt, n_inpt, n_latent, n_output,
                 make_recog, make_prior, make_gen, make_cond=None,
                 declare=None, name=None):
        self.inpt = inpt
        self.n_inpt = n_inpt
        self.n_latent = n_latent
        self.n_output = n_output
        self.make_recog = make_recog
        self.make_prior = make_prior
        self.make_gen = make_gen
        self.make_cond = make_cond
        self.transfer = _transfer
        super(VariationalAutoEncoder, self).__init__(
            declare=declare, name=name)

    def _forward(self):
        self.recog = self.make_recog(self.inpt)
        self.latent = self.recog.stt
        self.recog_sample = self.recog.sample()

        self.prior = self.make_prior(self.recog_sample)

        if self.make_cond is None:
            gen_inpt = self.recog_sample
        else:
            self.condition = self.make_cond(self.inpt)
            gen_inpt = T.concatenate(
                [self.recog_sample, self.condition], axis=self.latent.ndim - 1)

        # Generative model
        self.gen = self.make_gen(gen_inpt)
        self.gen_sample = self.gen.sample()
        self.output = self.gen.stt

    # TODO this is a pretty ugly hack to make things picklable.
    def __getstate__(self):
        state = self.__dict__.copy()
        unpicklables = 'make_gen make_recog make_prior make_cond'.split()
        for i in unpicklables:
            if i in state:
                del state[i]
