# -*- coding: utf-8 -*-

import signal
import time

import numpy as np

from climin import mathadapt as ma
from climin.util import iter_minibatches


class Trainer(object):

    def __init__(self, model, *args, **kwargs):
        self.model = model


class GentleTrainer(Trainer):

    # TODO This trainer needs a better name and documentation

    def __init__(self, ident, model, max_samples, sample_dims, log_func):
        self.ident = ident
        self.max_samples = max_samples
        self.sample_dims = sample_dims
        self.log_func = log_func

        self.best_pars = None
        self.best_loss = float('inf')
        self.infos = []

        self.stop_next = False
        signal.signal(signal.SIGINT, self._ctrl_c_handler)

        super(GentleTrainer, self).__init__(model)

    def minibatch_score(self, f, data):
        batches = iter_minibatches(data, self.max_samples, self.sample_dims, 1)
        score = 0.
        seen_samples = 0.
        for batch in batches:
            this_samples = batch[0].shape[self.sample_dims[0]]
            score += f(*batch) * this_samples
            seen_samples += this_samples
        return ma.scalar(score / seen_samples)

    def fit(self, fit_data, eval_data, val_key, stop, report):
        self.start = time.time()
        for info in self.model.iter_fit(*fit_data):
            if report(info):
                for key, data in eval_data.items():
                    info['%s_loss' % key] = self.minibatch_score(
                        self.model.score, data)

                if info['%s_loss' % val_key] < self.best_loss:
                    self.best_loss = info['val_loss']
                    self.best_pars = self.model.parameters.data.copy()

                info['best_loss'] = self.best_loss
                info['best_pars'] = self.best_pars

                info.update({
                    'time': time.time() - self.start,
                })

                # TODO We need a way to test a variable for being a gnumpy
                # array or a numpy array without gnumpy available.

                filtered_info = dict(
                    (k, v) for k, v in info.items()
                    #if (not isinstance(v, (np.ndarray, gp.garray)) or v.size <= 1) and k not in ('args', 'kwargs'))
                    if (not isinstance(v, (np.ndarray, )) or v.size <= 1) and k not in ('args', 'kwargs'))

                for key in filtered_info:
                    if isinstance(filtered_info[key], np.float32):
                        filtered_info[key] = float(filtered_info[key])
                self.log_func(self.ident, filtered_info)
                self.infos.append(filtered_info)

                if stop(info) or self.stop_next:
                    break

    def _ctrl_c_handler(self, signal, frame):
        self.stop_next = True
