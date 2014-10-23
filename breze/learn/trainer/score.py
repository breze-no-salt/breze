# -*- coding: utf-8 -*-

"""Module for various scoring strategies."""


from climin.util import iter_minibatches
from climin import mathadapt as ma


def simple(f_score, *data):
    """Simple scoring strategy which just applies ``f_score`` to the passed
    arguments."""
    return f_score(*data)


class MinibatchScore(object):
    """MinibatchScore class.

    Scoring strategy for very large data sets, where the score of only a subset
    of rows can be calculated at the same time. This score assumes that scores
    are averages.


    Attributes
    ----------

    max_samples : int
        Maximum samples to calculcate the score for at the same time.

    sample_dims : list of ints
        Dimensions along which the samples are stored. The length of this list\
        corresponds to the number of arguments the score takes. The entry along\
        which different samples are stored.
    """

    def __init__(self, max_samples, sample_dims):
        """Create MinibatchScore object.


        Parameters
        ----------

        max_samples : int
            Maximum samples to calculcate the score for at the same time.

        sample_dims : list of ints
            Dimensions along which the samples are stored. The length of this
            list corresponds to the number of arguments the score takes. The
            entry along which different samples are stored.
        """
        self.max_samples = max_samples
        self.sample_dims = sample_dims

    def __call__(self, f_score, *data):
        """"Return the score of the data."""
        batches = iter_minibatches(data, self.max_samples, self.sample_dims, 1)
        score = 0.
        seen_samples = 0.
        for batch in batches:
            this_samples = batch[0].shape[self.sample_dims[0]]
            score += f_score(*batch) * this_samples
            seen_samples += this_samples
        return ma.scalar(score / seen_samples)
