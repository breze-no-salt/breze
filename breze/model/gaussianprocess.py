# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from theano.sandbox.linalg.ops import MatrixInverse, Det, psd
minv = MatrixInverse()
det = Det()

from ..util import ParameterSet, Model, lookup
from ..component import distance


def softabs(X):
    return T.sqrt(X**2 + 1e-8)


def linear_kernel(X, X_, length_scales):
    length_scales = softabs(length_scales)
    X *= length_scales.dimshuffle('x', 0)
    X_ *= length_scales.dimshuffle('x', 0)
    return T.dot(X, X_.T)


def matern52_kernel(X, X_, length_scales):
    length_scales = softabs(length_scales)
    X = X * length_scales.dimshuffle('x', 0)
    X_ = X_ * length_scales.dimshuffle('x', 0)
    D = distance.distance_matrix(X, X_, 'soft_l1')
    return (1.0 + T.sqrt(5.) * D + (5. / 3.) * D**2) * T.exp(-T.sqrt(5.) * D)


def rbf_kernel(X, X_, length_scales):
    length_scales = softabs(length_scales)
    X = X * length_scales.dimshuffle('x', 0)
    X_ = X_ * length_scales.dimshuffle('x', 0)
    D = distance.distance_matrix(X, X_, 'soft_l1')
    return T.exp(-D**2)


class GaussianProcess(Model):

    minimal_noise = 1e-4
    maximal_length_scales = 10

    def __init__(self, n_inpt, kernel='linear'):
        self.n_inpt = n_inpt
        self.kernel = kernel
        self.f_predict = None

        super(GaussianProcess, self).__init__()
        self.parameters.data[:] = np.random.normal(0, 1e-1, self.parameters.data.shape)

    def init_pars(self):
        parspec = self.get_parameter_spec(self.n_inpt)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        self.exprs = self.make_exprs(
            T.matrix('inpt'), T.matrix('test_inpt'),
            T.matrix('target'),
            self.parameters.length_scales, self.parameters.noise,
            self.kernel)

    @staticmethod
    def get_parameter_spec(n_inpt):
        return dict(length_scales=n_inpt, noise=1)

    @staticmethod
    def make_exprs(inpt, test_inpt, target,
                   length_scales, noise, kernel):
        noise = T.maximum(softabs(noise), GaussianProcess.minimal_noise)
        length_scales = T.minimum(softabs(length_scales),
                                  GaussianProcess.maximal_length_scales)

        kernel_func = globals()['%s_kernel' % kernel]

        # For training.
        K = kernel_func(inpt, inpt, length_scales)

        # We add the noise variable, but at least 1e-4 for
        # numerical stability.
        K += T.identity_like(K) * noise
        # Justin: This is an informed choice. I played around a little
        # with various methods (e.g. using cholesky first) and came to
        # the conclusion that this way of doing it was way faster than
        # anything else in my case.
        psd(K)
        inv_K = minv(K)

        nll = 0.5 * (
            det(K)
            + T.dot(T.dot(target.T, inv_K), target).mean()
            + K.shape[0] * T.log(2 * np.pi))

        # For prediction.
        inference_kernelrows = kernel_func(inpt, test_inpt, length_scales)
        test_K = kernel_func(test_inpt, test_inpt, length_scales)

        kTK = T.dot(inference_kernelrows.T, inv_K)
        output_mean = T.dot(kTK, target)

        kTKk = T.dot(kTK, inference_kernelrows)

        d = test_inpt.shape[0]
        diag_coords = T.arange(d), T.arange(d)
        output_std = (test_K - kTKk)[diag_coords]
        return {
            'inpt': inpt,
            'test_inpt': test_inpt,
            'target': target,
            'gram_matrix': K,
            'nll': nll,
            'loss': nll,
            'output': output_mean,
            'output_std': output_std,
            'test_K': test_K,
            'kTKk': kTKk,
        }
