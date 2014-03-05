# -*- coding: utf-8 -*-

"""Module containing several losses usable for supervised and unsupervised
training. This is different from ``breze.component.loss`` in the sense that
each prediction is also assumed to have a variance.

The losses in this module assume two inputs: a target and a prediction.
Additionally, if the target has a dimensionality of D, the prediction is
assumed to have a dimensionality of 2D. The first D element constitute to the
mean while the latter to the variance.

Additionally, all losses from ``breze.arch.component.loss`` are also available;
here, we just ignore the variance part of the input to the loss.
"""

import numpy as np

import theano.tensor as T

import transfer
from breze.arch.component import loss


def unpack_mean_var(arr):
    """Given an array which containts the mean in the first and the variance in
    the second half in the second dimension, return two separate arrays.

    Parameters
    ----------

    arr : 2D Theano tensor, 2D array_like, 3D Theano Tensor, 3D array_like
        Array that contains the mean along the first and the variance along the
        second half of the last dimension.

    Returns
    -------

    mean : 3D theano tensor or 3D array_like

    var : 3D theano tensor or 3D array_like
    """
    if arr.ndim == 2:
        mean = arr[:, :arr.shape[1] // 2]
        var = arr[:, arr.shape[1] // 2:]
    elif arr.ndim == 3:
        mean = arr[:, :, :arr.shape[2] // 2]
        var = arr[:, :, arr.shape[2] // 2:]
    return mean, var


def discard_var_loss(loss):
    def inner(target, prediction):
        mean, var = unpack_mean_var(prediction)
        return loss(target, mean)
    return inner


squared = discard_var_loss(loss.squared)
absolute = discard_var_loss(loss.absolute)
multinomial_ce = discard_var_loss(loss.multinomial_ce)
nmultinomial_ce = discard_var_loss(loss.nmultinomial_ce)
bernoulli_ces = discard_var_loss(loss.bernoulli_ces)
ncac = discard_var_loss(loss.ncac)
ncar = discard_var_loss(loss.ncar)


# TODO: document
def make_expected_hinge(margin):
    def expected_hinge(target, prediction):
        target = 2 * target - 1
        pred_mean, pred_var = unpack_mean_var(prediction)
        mean, _ = transfer.rectifier(margin - target * pred_mean, pred_var)
        return mean
    return expected_hinge

expected_hinge_1 = make_expected_hinge(1)


# TODO: document
def make_expected_squared_hinge(margin):
    def expected_squared_hinge(target, prediction):
        target = 2 * target - 1
        pred_mean, pred_var = unpack_mean_var(prediction)

        mean_unsqrd, var_unsqrd = transfer.rectifier(
            margin - target * pred_mean, pred_var)
        # See Murphy, "Machine Learning: A Probabilist Perspective", Eq (2.26)
        # for the following step.
        mean = var_unsqrd + mean_unsqrd ** 2

        return mean
    return expected_squared_hinge

expected_squared_hinge_1 = make_expected_squared_hinge(1)


# TODO document
def diag_gaussian_nll(target, prediction):
    mean, var = unpack_mean_var(prediction)
    residuals = target - mean
    weighted_squares = -(residuals ** 2) / (2 * var + 1e-4)
    normalization = T.log(T.sqrt(2 * np.pi * var + 1e-4))
    ll = weighted_squares - normalization
    return -ll
