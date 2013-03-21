# -*- coding: utf-8 -*-


import itertools

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from brummlearn.gaussianprocess import GaussianProcess

from .acquisition import expected_improvement

import climin
import climin.stops


class Searcher(object):

    def __init__(self, size):
        self.size = size
        self.handles = itertools.count()
        self.results = []
        self.candidates = {}

    def push_result(self, handle, result):
        self.results.append((handle, self.candidates[handle], result))


class RandomSearcher(Searcher):

    def pull_candidate(self):
        handle, point = self.handles.next(), np.random.random(self.size)
        self.candidates[handle] = point
        return handle, point


class ModelBasedSearcher(RandomSearcher):

    def __init__(self, size, n_candidates=1000, initial_random_tries=10):
        self.n_candidates = n_candidates
        self.initial_random_tries = initial_random_tries
        self.model = None

        super(ModelBasedSearcher, self).__init__(size)

    def _data_sets(self):
        X = np.empty((len(self.results), self.size))
        Z = np.empty((len(self.results)))

        for i, (_, x, z) in enumerate(self.results):
            X[i] = x
            Z[i] = z

        return X, Z

    def pull_candidate(self):
        # Do random search for the first few tries.
        if len(self.results) < self.initial_random_tries:
            return RandomSearcher.pull_candidate(self)

        # Create function that measures how good it is to try out a solution.
        best = min(i for _, _, i in self.results)

        # Randomly sample a set of candidates.
        candidates = np.random.random((self.n_candidates, self.size))

        # Determine the acquisition values.
        acquisitions = self.expected_improvement(candidates, best)

        # Return the best of those candidates according to the acquisition
        # function.
        point = candidates[acquisitions.argmin()]

        handle = self.handles.next()
        self.candidates[handle] = point
        return handle, point

    ei_plots = itertools.count()

    def expected_improvement(self, candidates, best):
        f_model_cost = self._fit_model_cost()
        mean, var = f_model_cost(candidates)
        return expected_improvement(mean, var, best)


class GaussianProcessSearcher(ModelBasedSearcher):

    def _fit_model_cost(self):
        X, Z = self._data_sets()
        Z = Z[:, np.newaxis]

        if self.model is None:
            self.model = GaussianProcess(
                X.shape[1], kernel='matern52', optimizer='lbfgs', max_iter=100)
            self.model.parameters.data[:] = np.log(1.)
            self.f_nll = self.model.function(['inpt', 'target'], 'nll')

        stop = climin.stops.any_([
            climin.stops.converged(lambda: self.f_nll(X, Z),
                                   epsilon=1e-3, patience=5),
            climin.stops.after_n_iterations(100)])
        for i, info in enumerate(self.model.iter_fit(X, Z)):
            if stop(info):
                break

        f_model_cost = lambda x: self.model.predict(x, True)
        return f_model_cost


class RandomForestSearcher(ModelBasedSearcher):

    def _fit_model_cost(self, X, Z):
        X, Z = self._data_sets()
        model = self.model = RandomForestRegressor()
        model.fit(X, Z)

        def f_model_cost(x):
            predictions = np.array([i.predict(x) for i in model.estimators_])
            return predictions.mean(), [predictions.std()**2]

        return f_model_cost


class BayesianGaussianProcessSearcher(GaussianProcessSearcher):

    def __init__(self, size, n_candidates=1000, initial_random_tries=10,
                 n_samples=10):
        super(BayesianGaussianProcessSearcher, self).__init__(
            size, n_candidates, initial_random_tries)
        self.n_samples = n_samples

    def expected_improvement(self, candidates, best):
        ei = np.empty((self.n_samples, candidates.shape[0]))
        m = np.empty((self.n_samples, candidates.shape[0]))
        v = np.empty((self.n_samples, candidates.shape[0]))

        f_model_cost = self._fit_model_cost()

        for i in range(self.n_samples):
            self.model.sample_parameters()
            m, v = f_model_cost(candidates)
            ei[i] = expected_improvement(m[:, 0], v[:, 0], best)
        return ei.mean(axis=0)
