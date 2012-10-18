# -*- coding: utf-8 -*-


import itertools

import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn.ensemble import RandomForestRegressor

from .acquisition import expected_improvement


class Searcher(object):

    def __init__(self, size):
        self.size = size
        self.handles = itertools.count()
        self.results = []
        self.candidates = {}


class RandomSearcher(Searcher):

    def pull_candidate(self):
        handle, point = self.handles.next(), np.random.random(self.size)
        self.candidates[handle] = point
        return handle, point

    def push_result(self, handle, result):
        self.results.append((handle, self.candidates[handle], result))



class ModelBasedSearcher(RandomSearcher):

    def __init__(self, size, n_candidates=1000, initial_random_tries=10):
        self.n_candidates = n_candidates
        self.initial_random_tries = initial_random_tries

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

        # Create data set.
        X, Z = self._data_sets()

        # Create model and fit it to get a p(c|x) defined by its mean and its
        # std.
        f_model_cost = self._fit_model_cost(X, Z)

        # Create function that measures how good it is to try out a solution.
        best = min(i for _, _, i in self.results)
        acq_func = expected_improvement(f_model_cost, best)

        # Randomly sample a set of candidates.
        candidates = np.random.random((self.n_candidates, self.size))

        # Return the best of those candidates according to the acquisition
        # function.
        point = candidates[acq_func(candidates).argmin()]

        handle = self.handles.next()
        self.candidates[handle] = point
        return handle, point


class GaussianProcessSearcher(ModelBasedSearcher):

    def __init__(self, size, n_candidates=1000, initial_random_tries=10):
        super(GaussianProcessSearcher, self).__init__(
            size, n_candidates, initial_random_tries)
        self.theta0 = .1

    def _fit_model_cost(self, X, Z):
        while True:
            model = GaussianProcess(
                theta0=self.theta0, thetaL=.1, thetaU=2., nugget=0.01)
            try:
                model.fit(X, Z)
            except Exception:
                self.theta0 *= 1.5
                continue
            break
        f_model_cost = lambda x: model.predict(x, eval_MSE=True)
        return f_model_cost


class RandomForestSearcher(ModelBasedSearcher):

    def _fit_model_cost(self, X, Z):
        model = RandomForestRegressor()
        model.fit(X, Z)

        def f_model_cost(x):
            predictions = np.array([i.predict(x) for i in model.estimators_])
            return predictions.mean(), [predictions.std()**2]

        return f_model_cost
