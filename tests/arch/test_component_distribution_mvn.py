# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

from breze.arch.component.distributions import mvn


def test_pdf_compare_logpdf():
    theano.config.compute_test_value = 'raise'
    sample = T.matrix()
    sample.tag.test_value = np.random.random((10, 5)).astype(theano.config.floatX)

    mean = T.vector()
    mean.tag.test_value = np.empty(5).astype(theano.config.floatX)

    cov = T.matrix()

    r = np.random.random((5, 5))
    c = np.dot(r, r.T) + np.eye(5)
    cov.tag.test_value = c.astype(theano.config.floatX)

    density = mvn.pdf(sample, mean, cov)
    log_density = mvn.logpdf(sample, mean, cov)

    f_density = theano.function([sample, mean, cov], density)
    f_logdensity = theano.function([sample, mean, cov], log_density)

    some_sample = np.random.random((20, 5)).astype(theano.config.floatX)
    some_mean = np.array([1., 2., 3., 4., 5.]).astype(theano.config.floatX)
    w = np.random.random((5, 5)).astype(theano.config.floatX)

    some_cov = np.dot(w, w.T) + np.eye(5).astype(theano.config.floatX)

    d = f_density(some_sample, some_mean, some_cov)
    log_d = f_logdensity(some_sample, some_mean, some_cov)

    print np.log(d) / log_d

    assert np.allclose(np.log(d), log_d), '%s %s' % (np.log(d), log_d)
