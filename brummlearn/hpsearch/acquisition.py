# -*- coding: utf-8 -*-


import numpy as np
from scipy import stats


def expected_improvement(f_model_cost, best_loss, atleast=0.00):
    def inner(x):
        mean, var = f_model_cost(x)
        if var[0] == 0:
            return np.zeros(x.shape)
        std = np.sqrt(var)
        improv = mean - best_loss + atleast
        z = improv / std
        return (improv * stats.norm.cdf(z) + std * stats.norm.pdf(z))
    return inner
