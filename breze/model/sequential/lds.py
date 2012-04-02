# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.linalg import det, matrix_inverse

from ...util import ParameterSet, Model, lookup


def stacked_dot(X, Y):
    # Only works for matrices! not for vectors.
    if X.ndim == 2 and Y.ndim == 2:
        res = T.dot(X, Y)
    elif X.ndim == 3 and Y.ndim == 2:
        X_ = X.dimshuffle(0, 1, 2, 'x')
        Y_ = Y.dimshuffle('x', 'x', 0, 1)
        res = (X_ * Y_).sum(axis=2)
    elif X.ndim == 2 and Y.ndim == 3:
        X_ = X.dimshuffle('x',  0, 1, 'x')
        Y_ = Y.dimshuffle(0, 'x', 1, 2)
        res = (X_ * Y_).sum(axis=2)
    elif X.ndim == 3 and Y.ndim == 3:
        X_ = X.dimshuffle(0, 1, 2, 'x')
        Y_ = Y.dimshuffle(0, 'x', 1, 2)
        res = (X_ * Y_).sum(axis=2)
    else:
        raise ValueError('dimensions wrong: %i and %i' % (X.ndim, Y.ndim))
    return res


def filter_and_prob(inpt, transition, emission,
           visible_noise_mean, visible_noise_cov,
           hidden_noise_mean, hidden_noise_cov,
           initial_hidden, initial_hidden_cov):
    step = forward_step(
        transition, emission,
        visible_noise_mean, visible_noise_cov,
        hidden_noise_mean, hidden_noise_cov)

    # Make sure we have a initial hidden and initial covariance for
    # each sequence.
    ones = T.ones_like(inpt[0, :, 0])
    initial_hidden = T.outer(ones, initial_hidden)
    initial_hidden_cov_flat = initial_hidden_cov.flatten()
    initial_hidden_cov = T.outer(ones, initial_hidden_cov_flat
                  ).reshape((inpt.shape[1],
                             initial_hidden.shape[1],
                             initial_hidden.shape[1]))

    (f, F, log_p), _ = theano.scan(
        step,
        sequences=inpt,
        outputs_info=[initial_hidden, initial_hidden_cov, None])

    return f, F, log_p


def smooth(filtered_means, filtered_covs, transition,
           hidden_noise_mean, hidden_noise_cov):
    step = backward_step(transition, hidden_noise_mean, hidden_noise_cov)

    (g, G), _ = theano.scan(
        step,
        sequences=[filtered_means[:-1], filtered_covs[:-1]],
        outputs_info=[filtered_means[-1], filtered_covs[-1]],
        go_backwards=True)

    return g, G


def forward_step(transition, emission,
                 visible_noise_mean, visible_noise_cov,
                 hidden_noise_mean, hidden_noise_cov):
    # The crux about the following code is that we perform several
    # filtering operations in parallel--in order to be more efficient
    # by using batches. To keep things understandable, each major
    # calculation has a comment indicating the shape of the resulting
    # tensor. In these, `n` stands for the number of sequences that is
    # being looked at, `h` for the dimensionality of the latent space and
    # v for the dimensionality of the visible space.
    vnm, vnc = visible_noise_mean, visible_noise_cov
    hnm, hnc = hidden_noise_mean, hidden_noise_cov

    def step(visible, filtered_hidden_mean_m1, filtered_hidden_cov_m1):
        A, B = transition, emission                             # (h, h), (h, d)
        # Shortcuts for the filtered mean and covariance from the previous
        # time step.
        f_m1 = filtered_hidden_mean_m1                      # (n, h)
        F_m1 = filtered_hidden_cov_m1                       # (n, h, h)

        # Calculate mean of joint.
        hidden_mean = T.dot(f_m1, A) + hnm                  # (n, h)

        visible_mean = T.dot(hidden_mean, B) + vnm          # (n, v)

        # Calculate covariance of joint.
        hidden_cov = stacked_dot(
            A.T, stacked_dot(F_m1, A))                      # (n, h, h)
        hidden_cov +=  hnc

        visible_cov = stacked_dot(                          # (n, v, v)
            B.T, stacked_dot(hidden_cov, B))
        visible_cov += vnc

        visible_hidden_cov = stacked_dot(hidden_cov, B)     # (n, h, v)

        visible_error = visible - visible_mean              # (n, v)

        inv_visible_cov, _ = theano.map(
            lambda x: matrix_inverse(x), visible_cov)       # (n, v, v)

        # I don't know a better name for this monster.
        visible_hidden_cov_T = visible_hidden_cov.dimshuffle(0, 2, 1)
        D = stacked_dot(visible_hidden_cov_T,
                        inv_visible_cov)                    # (n, h, v)

        f = (D * visible_error.dimshuffle(0, 'x', 1)        # (n, h)
            ).sum(axis=2)
        f += hidden_mean

        F = hidden_cov
        F -= stacked_dot(D, visible_hidden_cov)

        log_p = (inv_visible_cov *                          # (n,)
                 visible_error.dimshuffle(0, 1, 'x') *
                 visible_error.dimshuffle(0,'x', 1)).sum(axis=(1, 2))
        dets, _ = theano.map(
            lambda x: det(x), 2 * np.pi * visible_cov)
        log_p /= T.sqrt(dets)

        return f, F, log_p

    return step


def backward_step(transition, hidden_noise_mean, hidden_noise_cov):
    A = transition
    def step(filtered_mean, filtered_cov,
             smoothed_mean_p1, smoothed_cov_p1):
        f, F = filtered_mean, filtered_cov                  # (n, h), (n, h, h)

        # Statistics for p(h, h_m1 | V)
        hidden_mean = T.dot(f, A) + hidden_noise_mean       # (n, h)
        hidden_cov = stacked_dot(A.T,
                                 stacked_dot(F, A))         # (n, h, h)
        hidden_cov += hidden_noise_cov

        inv_hidden_cov, _ = theano.map(
            lambda x: matrix_inverse(x), hidden_cov)        # (n, h, h)

        hidden_p1_hidden_cov = stacked_dot(A, F)            # (n, h, h)
        hidden_p1_hidden_cov_T = hidden_p1_hidden_cov.dimshuffle(0, 2, 1)

        cov_rev = F - stacked_dot(
            stacked_dot(hidden_p1_hidden_cov_T, inv_hidden_cov),
            hidden_p1_hidden_cov)                           # (n, h, h)


        trans_rev = stacked_dot(hidden_p1_hidden_cov_T,     # (n, h, h)
                                inv_hidden_cov)

        mean_rev = f
        mean_rev -= (hidden_mean.dimshuffle(0, 'x', 1) * trans_rev # (n, h)
                    ).sum(axis=2)

        # Turn these into matrices so they work with stacked_dot.
        smoothed_mean_p1 = smoothed_mean_p1.dimshuffle(0, 1, 'x')

        smoothed_mean = stacked_dot(smoothed_mean_p1, trans_rev)
        smoothed_mean = smoothed_mean[:, :, 0]
        smoothed_mean += mean_rev

        smoothed_cov = stacked_dot(trans_rev,
                                   stacked_dot(smoothed_cov_p1, trans_rev))

        smoothed_cov += cov_rev

        return smoothed_mean, smoothed_cov
    return step


class LinearDynamicalSystem(Model):

    def __init__(self, n_inpt, n_hidden):
        self.n_inpt = n_inpt
        self.n_hidden = n_hidden

        super(LinearDynamicalSystem, self).__init__()

    def init_pars(self):
        parspec = self.get_parameter_spec(self.n_inpt, self.n_hidden)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        self.exprs = self.make_exprs(
            T.tensor3('inpt'),
            self.parameters.transition,
            self.parameters.emission,
            self.parameters.visible_noise_mean,
            self.parameters.visible_noise_cov,
            self.parameters.hidden_noise_mean,
            self.parameters.hidden_noise_cov,
            self.parameters.hidden_mean_initial,
            self.parameters.hidden_cov_initial)


    @staticmethod
    def get_parameter_spec(n_inpt, n_hidden):
        return dict(
            transition=(n_hidden, n_hidden),
            emission=(n_hidden, n_inpt),
            visible_noise_mean=n_inpt,
            visible_noise_cov=(n_inpt, n_inpt),
            hidden_noise_mean=n_hidden,
            hidden_noise_cov=(n_hidden, n_hidden),
            hidden_mean_initial=n_hidden,
            hidden_cov_initial=(n_hidden, n_hidden))

    @staticmethod
    def make_exprs(inpt, transition, emission, 
                   visible_noise_mean, visible_noise_cov,
                   hidden_noise_mean, hidden_noise_cov,
                   hidden_mean_initial, hidden_cov_initial):
        filtered_means, filtered_covs, prob = filter_and_prob(
            inpt, transition, emission,
            visible_noise_mean, visible_noise_cov,
            hidden_noise_mean, hidden_noise_cov,
            hidden_mean_initial, hidden_cov_initial)
        smoothed_means, smoothed_covs = smooth(
            filtered_means, filtered_covs, transition, 
            hidden_noise_mean, hidden_noise_cov)

        return {
            'inpt': inpt,
            'filtered_means': filtered_means,
            'filtered_covs': filtered_covs,
            'smoothed_means': smoothed_means,
            'smoothed_covs': smoothed_covs,
            'log_likelihood': prob
        }





























