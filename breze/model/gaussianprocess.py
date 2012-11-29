# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from theano.sandbox.linalg.ops import MatrixInverse, Det, psd, Cholesky
minv = MatrixInverse()
det = Det()
cholesky = Cholesky()

from ..util import ParameterSet, Model
from ..component import misc


def linear_kernel(X, X_, length_scales, amplitude, diag=False):
    X = X * length_scales.dimshuffle('x', 0)
    X_ = X_ * length_scales.dimshuffle('x', 0)
    if diag:
        return amplitude * (X * X_).sum(axis=1)
    else:
        return amplitude * T.dot(X, X_.T)


def matern52_kernel(X, X_, length_scales, amplitude, diag=False):
    X = X * length_scales.dimshuffle('x', 0)
    X_ = X_ * length_scales.dimshuffle('x', 0)
    if not diag:
        D2 = misc.distance_matrix(X, X_, 'l2')
    else:
        D2 = ((X - X_)**2).sum(axis=1)
    D = T.sqrt(D2 + 1e-8)
    return amplitude * (1.0 + T.sqrt(5.) * D + (5. / 3.) * D2) * T.exp(-T.sqrt(5.) * D)


def rbf_kernel(X, X_, length_scales, amplitude, diag=False):
    X = X * length_scales.dimshuffle('x', 0)
    X_ = X_ * length_scales.dimshuffle('x', 0)
    if not diag:
        D2 = misc.distance_matrix(X, X_, 'l2')
    else:
        D2 = ((X - X_)**2).sum(axis=1)

    return amplitude * T.exp(-D2)


class GaussianProcess(Model):

    minimal_noise = 1e-4
    minimal_length_scale = 1e-8

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
            self.parameters.amplitude,
            self.kernel)

    @staticmethod
    def get_parameter_spec(n_inpt):
        return dict(length_scales=n_inpt, noise=1, amplitude=1)

    @staticmethod
    def make_exprs(inpt, test_inpt, target,
                   length_scales, noise, amplitude, kernel):

        # To stay compatible to the prediction api, target will
        # be a matrix to the outside. But in the following, it's easier
        # if it is a vector in the inside. We keep a reference to the
        # matrix anyway, to return it.
        target_ = target
        target = target[:, 0]
        noise = T.exp(noise) + GaussianProcess.minimal_noise
        length_scales = T.exp(length_scales) + GaussianProcess.minimal_length_scale
        amplitude = T.exp(amplitude) + 1e-4

        if isinstance(kernel, (unicode, str)):
            kernel_func = globals()['%s_kernel' % kernel]
        else:
            kernel_func = kernel

        # For training.
        K = kernel_func(inpt, inpt, length_scales, amplitude)

        K += T.identity_like(K) * noise

        # Justin: This is an informed choice. I played around a little
        # with various methods (e.g. using cholesky first) and came to
        # the conclusion that this way of doing it was way faster than
        # anything else in my case.
        psd(K)
        inv_K = minv(K)

        n_samples = K.shape[0]
        nll = (
            0.5 * T.dot(T.dot(target.T, inv_K), target)
            + 0.5 * T.log(det(K) + 1e-8)
            + 0.5 * n_samples * T.log(2 * np.pi))

        # For prediction.
        inference_kernelrows = kernel_func(inpt, test_inpt, length_scales,
                                           amplitude)

        kTK = T.dot(inference_kernelrows.T, inv_K)
        output_mean = T.dot(kTK, target).dimshuffle(0, 'x')

        kTKk = T.dot(kTK, inference_kernelrows)

        chol_inv_K = cholesky(inv_K)

        diag_kTKk = (T.dot(chol_inv_K.T, inference_kernelrows)**2).sum(axis=0)
        test_K = kernel_func(test_inpt, test_inpt, length_scales, amplitude,
                             diag=True)
        output_var = ((test_K - diag_kTKk)).dimshuffle(0, 'x')

        return {
            'inpt': inpt,
            'test_inpt': test_inpt,
            'target': target_,
            'gram_matrix': K,
            'inv_gram_matrix': inv_K,
            'chol_inv_gram_matrix': chol_inv_K,
            'nll': nll,
            'loss': nll,
            'output': output_mean,
            'output_var': output_var,
            'inference_kernelrows': inference_kernelrows,
            'test_K': test_K,
            'kTKk': kTKk,
        }
