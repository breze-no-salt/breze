# -*- coding: utf-8 -*-


import numpy as np
from scipy import stats


def expected_improvement(means, variances, best_loss, atleast=0.00):
    std = np.sqrt(variances + 1e-8)
    improv = means - best_loss + atleast
    z = improv / std
    return (improv * stats.norm.cdf(z) + std * stats.norm.pdf(z))
