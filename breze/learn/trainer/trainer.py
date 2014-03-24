# -*- coding: utf-8 -*-

# TODO document
import signal
import time
import types

import numpy as np
from climin import mathadapt as ma
from copy import deepcopy

import score as score_


class Trainer(object):

    def __init__(self, ident, model, score=score_.simple, *args, **kwargs):
        self.ident = ident
        self.model = model
        self._score = score
        self.best_pars = None
        self.best_loss = float('inf')
        self.infos = []
        self.current_info = None

    def stop(self, stop_info):
        return stop_info

    def score(self, *data):
        return self._score(self.model.score, *data)

    def handle_update(self, fit_data, eval_data):
        update_losses = {
            'loss': ma.scalar(self.score(*fit_data))
        }
        for key, data in eval_data.items():
            update_losses['%s_loss' % key] = self.score(*data)
        return update_losses

    def fit(self, fit_data, eval_data, stop, report, val_key='val'):
        start = time.time()
        for info in self.model.iter_fit(*fit_data, info_opt=self.current_info):
            if report(info):
                update_losses = self.handle_update(fit_data, eval_data)
                for key, data in update_losses.items():
                    info[key] = data

                if info['%s_loss' % val_key] < self.best_loss:
                    self.best_loss = info['%s_loss' % val_key]
                    self.best_pars = self.model.parameters.data.copy()

                info['best_loss'] = self.best_loss
                info['best_pars'] = self.best_pars

                info.update({
                    'time': time.time() - start,
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

                self.infos.append(filtered_info)
                self.current_info = info
                yield info

                if self.stop(stop(info)):
                    break
