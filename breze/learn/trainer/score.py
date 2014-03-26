# -*- coding: utf-8 -*-


from climin.util import iter_minibatches
from climin import mathadapt as ma


def simple(f_score, *data):
    return f_score(*data)


class MinibatchScore(object):

    def __init__(self, max_samples, sample_dims):
        self.max_samples = max_samples
        self.sample_dims = sample_dims

    def __call__(self, f_score, *data):
        batches = iter_minibatches(data, self.max_samples, self.sample_dims, 1)
        score = 0.
        seen_samples = 0.
        for batch in batches:
            this_samples = batch[0].shape[self.sample_dims[0]]
            score += f_score(*batch) * this_samples
            seen_samples += this_samples
        return ma.scalar(score / seen_samples)
