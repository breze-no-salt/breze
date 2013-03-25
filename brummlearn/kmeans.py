# -*- coding: utf-8 -*-

import itertools

import numpy as np
from sklearn.utils import check_random_state

from brummlearn.pca import Pca


class GainShapeKMeans(object):
    """GainShapeKMeans class to perform K-means clustering for feature learning as
    described in [LFRKM]_.


    Parameters
    ----------

    n_components : integer
        Number of features to learn.

    zscores : boolean, optional, default: False
        Flag indicating whether the data should be normalized to zero mean and
        unit variance before training and transformation.

    whiten : boolean, optional, default: False
        Flag indicating whether the data should be whitened before training and
        transformation.

    max_iter : integer, optional
        Maximum number of iterations to perform.

    random_state : None, integer or numpy.RandomState, optional, default: None
        Generator to initialize the dictionary. If None, the numpy singleton
        generator is used.


    Attributes
    ----------

    activation: {'identity', 'omp-1', 'soft-threshold'}, optional, default: None
        Activation to for transformation. 'identity' does not alter the output.
        'omp-1' only retains the component with the largest absolute value.
        'soft-threshold' only sets components below a certain threshold to
        zero, but separates positive and negative parts.

    threshold : scalar,
        Threshold used for soft-thresholding activation. Ignored if another
        activation is used.


    References
    ----------
    .. [LFRKM] `Learning Feature Representations with K-means`,
       Adam Coates (2012)

    """

    def __init__(self, n_component, zscores=False, whiten=False, max_iter=10,
                 random_state=None):
        self.n_component = n_component
        self.max_iter = max_iter
        self.random_state = random_state
        self.zscores = zscores
        self.whiten = whiten

        self.activation = 'identity'
        self.threshold = None

    def prepare(self, n_inpt):
        """Initialize the models internal structures.

        Parameters
        ----------

        n_inpt : integer
            Dimensionality of a single training examples.
        """
        self.random_state = check_random_state(self.random_state)
        self.dictionary = self.random_state.standard_normal(
            (n_inpt, self.n_component))
        self.normalize_dict()

    def normalize_dict(self):
        """Normalize the columns of the dictionary to unit length."""
        lengths = np.sqrt((self.dictionary ** 2).sum(axis=0))
        self.dictionary /= lengths

    def fit(self, X):
        """Fit the parameters of the model.

        Parameters
        ----------

        X : array_like
            Array of shape ``(n_samples, n_inpt)`` used for training."""
        if self.zscores:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            X -= self.mean
            X /= self.std
        if self.whiten:
            self.pca = Pca(whiten=True)
            self.pca.fit(X)
            X = self.pca.transform(X)
        self.prepare(X.shape[1])
        for i, info in enumerate(self.iter_fit(X)):
            if i + 1 >= self.max_iter:
                break

    def iter_fit(self, X):
        self.prepare(X.shape[1])
        for i in itertools.count():
            code = self.transform(X, activation='omp-1')
            self.dictionary += np.dot(X.T, code)
            self.normalize_dict()
            yield {'n_iter': i}

    def transform(self, X, activation=None):
        """Transform the data according to the dictionary.

        Parameters
        ----------

        X : array_like
            Input data of shape ``(n_samples, n_inpt)``.
        activation: {'identity', 'omp-1'}, optional, default: None
            Activation to use. 'linear' does not alter the output. 'omp-1'
            only retains the component with the largest absolute value.
            'soft-threshold' only sets components below a certain threshold to
            zero, but separates positive and negative parts. If None,
            ``.activation`` is used.

        """
        activation = self.activation if activation is None else activation
        if self.zscores:
            X -= self.mean
            X /= self.std
        if self.whiten:
            X = self.pca.transform(X)
        code = np.dot(X, self.dictionary)
        if activation == 'omp-1':
            mask = np.zeros(code.shape)
            mask[xrange(X.shape[0]), abs(code).argmax(axis=1)] = 1
            code *= mask
        elif activation == 'soft-threshold':
            positive = np.maximum(0, code - self.threshold)
            negative = np.maximum(0, -code - self.threshold)
            code = np.concatenate([positive, negative], axis=1)
        elif activation == 'identity':
            pass
        else:
            raise ValueError('unknown activation %s' % activation)
        return code
