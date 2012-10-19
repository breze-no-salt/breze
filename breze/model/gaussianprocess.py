# -*- coding: utf-8 -*-


import numpy as np
import theano.tensor as T

from theano.sandbox.linalg.ops import MatrixInverse, Det
minv = MatrixInverse()
det = Det()

from ..util import ParameterSet, Model, lookup
from ..component import transfer, distance



def linear_kernel(X, X_, length_scales):
    # Scale dimensions.
    length_scales = T.sqrt(length_scales**2 + 1e-8)
    X *= length_scales.dimshuffle('x', 0)
    X_ *= length_scales.dimshuffle('x', 0)
    return T.dot(X, X_.T)


class GaussianProcess(Model):

    def __init__(self, n_inpt, noise=1e-6):
        self.n_inpt = n_inpt
        self.noise = noise
        self.f_predict = None

        super(GaussianProcess, self).__init__()

        self.parameters.data[:] = np.random.normal(0, 1, self.parameters.data.shape)

    def init_pars(self):
        parspec = self.get_parameter_spec(self.n_inpt)
        self.parameters = ParameterSet(**parspec)

    def init_exprs(self):
        self.exprs = self.make_exprs(
            T.matrix('inpt'), T.matrix('test_inpt'),
            T.matrix('target'),
            self.parameters.length_scales, self.noise)

    @staticmethod
    def get_parameter_spec(n_inpt):
        return dict(length_scales=n_inpt)

    @staticmethod
    def make_exprs(inpt, test_inpt, target,
                   length_scales, noise):
        K = linear_kernel(inpt, inpt, length_scales)
        K += T.identity_like(K) * noise
        inv_K = minv(K)

        nll = 0.5 * (
               det(K)
               + T.dot(T.dot(target.T, inv_K), target).mean()
               + K.shape[0] * T.log(2 * np.pi))

        inference_kernelrows = linear_kernel(inpt, test_inpt, length_scales)

        kTK = T.dot(inference_kernelrows.T, inv_K)
        output_mean = T.dot(kTK, target)

        d = inpt.shape[1]
        diag_coords = T.arange(d), T.arange(d)
        c = K[diag_coords]
        output_std = c - T.dot(kTK, inference_kernelrows)[diag_coords]

        return {
            'inpt': inpt,
            'test_inpt': test_inpt,
            'target': target,
            'gram_matrix': K,
            'nll': nll,
            'loss': nll,
            'output': output_mean,
            'output_std': output_std,
        }
