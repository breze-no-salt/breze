# -*- coding: utf-8 -*-


import itertools

import numpy as np
from numpy.random import uniform


class Diverged(Exception):
    pass


def sample(f_ll, position, window_inc=1.0, max_widenings=1000):
    """Perform multivariate slice sampling.

    The idea is to randomly chose a direction and then perform univariate slice
    sampling along that direction.

    :param f_ll: Function that is proportional to the log likelihood of the
        model to sample from.
    :param position: One dimensional numpy array at which the sampling is
        started.
    :window_inc: Amount by which the window size is increased or decreased.
    :returns: Array of the same size as `position` which is a new sample.
    """
    direction = np.random.normal(0, 1, size=position.shape)
    direction /= np.sqrt((direction**2).sum())

    # Convenience function of the log likelihood along a direction.
    def ll_along_dir(step_size):
        return f_ll(direction * step_size + position)

    # Create initial bracket.
    upper = window_inc * uniform(0, 1)
    lower = upper - window_inc
    llh_0 = np.log(uniform(0, 1)) + ll_along_dir(0.0)

    for i in range(max_widenings):
        if ll_along_dir(lower) < llh_0:
            break
        lower -= window_inc
    for i in range(max_widenings):
        if ll_along_dir(upper) < llh_0:
            break
        upper += window_inc

    while True:
        step = uniform(lower, upper)
        llh = ll_along_dir(step)

        if np.isnan(llh):
            raise Diverged('log likelihood is NaN')

        if llh > llh_0:
            # Found good sample.
            break
        elif step < 0:
            lower = step
        elif step > 0:
            upper = step
        else:
            raise Diverged("slice has width 0")

    return step * direction + position
