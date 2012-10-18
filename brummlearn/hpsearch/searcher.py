# -*- coding: utf-8 -*-


import itertools

import numpy as np


class RandomSearcher(object):

    def __init__(self, size):
        self.size = size
        self.handles = itertools.count()
        self.results = []
        self.candidates = {}

    def pull_candidate(self):
        handle, point = self.handles.next(), np.random.random(self.size)
        self.candidates[handle] = point
        return handle, point

    def push_result(self, handle, result):
        self.results.append((handle, self.candidates[handle], result))
