# -*- coding: utf-8 -*-

import random

import numpy as np

from breze.learn.glm import GeneralizedLinearModel


def test_glm_fit():
    X = np.arange(-2, 2, .01)[:, np.newaxis]
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 200)
    X = X[idxs]
    Z = np.sin(X)

    glm = GeneralizedLinearModel(1, 1, max_iter=10)
    glm.fit(X, Z)


def test_glm_iter_fit():
    X = np.arange(-2, 2, .01)[:, np.newaxis]
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 200)
    X = X[idxs]
    Z = np.sin(X)

    glm = GeneralizedLinearModel(1, 1, max_iter=10)
    for i, info in enumerate(glm.iter_fit(X, Z)):
        if i >= 10:
            break


def test_glm_predict_linear():
    X = np.arange(-2, 2, .01)[:, np.newaxis]
    idxs = range(X.shape[0])
    idxs = random.sample(idxs, 200)
    X = X[idxs]
    Z = np.sin(X)

    glm = GeneralizedLinearModel(1, 1, max_iter=10)
    glm.fit(X, Z)
    glm.predict(X)

