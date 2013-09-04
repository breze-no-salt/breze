# -*- coding: utf-8 -*-


import numpy as np


class Numeric(object):

    def __init__(self, intify=False):
        self.intify = intify

    def transform(self, seed):
        res = self._transform(seed)
        if self.intify:
            res = int(round(res))
        return res


class Uniform(Numeric):

    def __init__(self, lower, upper, intify=False):
        self.size = 1
        self.lower = lower
        self.upper = upper
        super(Uniform, self).__init__(intify)

    def _transform(self, seed):
        # Pick only the first entry.
        seed = seed[0]
        scale = self.upper - self.lower
        return seed * scale + self.lower


class LogUniform(Numeric):

    def __init__(self, lower, upper, intify=False):
        self.size = 1
        self.lower = lower
        self.upper = upper
        self.uniform = Uniform(np.log(lower), np.log(upper))
        super(LogUniform, self).__init__(intify)

    def _transform(self, seed):
        x = self.uniform.transform(seed)
        return np.exp(x)


class OneOf(object):

    def __init__(self, choices):
        self.size = len(choices)
        self.choices = choices

    def transform(self, seed):
        return self.choices[seed.argmax()]


class SearchSpace(object):

    def __init__(self):
        self.variables = []

    def add(self, handle, var):
        self.variables.append((handle, var))

    @property
    def seed_size(self):
        return sum(i.size for _, i in self.variables)

    def transform(self, seed):
        """Transform a seed into a argument dictionary."""
        if seed.size != self.seed_size:
            raise ValueError('incorrect seed size, expected %i instead of %i'
                             % (self.seed_size, seed.size))
        start = 0
        sample = {}
        for handle, v in self.variables:
            stop = start + v.size
            sample[handle] = v.transform(seed[start:stop])
            start = stop
        return sample
