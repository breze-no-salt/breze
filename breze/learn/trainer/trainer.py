# -*- coding: utf-8 -*-

# TODO document
import signal
import time
import numpy as np
from climin import mathadapt as ma
from climin.util import iter_minibatches

class Trainer(object):

    def __init__(self, ident, model, *args, **kwargs):
        self.model = model
        self.ident = ident
        self.best_pars = None
        self.best_loss = float('inf')
        self.infos = []
        self.current_info = None

    def stop(self, stop_info):
        return stop_info

    def handle_update(self, fit_data, eval_data):
        update_losses = []
        update_losses['loss'] = ma.scalar(self.score(*fit_data))
        for key, data in eval_data.items():
            update_losses['%s_loss' % key] = self.score(data)
        return update_losses

    def fit(self, fit_data, eval_data, stop, report, val_key='val'):
        start = time.time()
        for info in self.model.iter_fit(*fit_data, info=self.current_info):
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

class GentleTrainer(Trainer):

    # TODO This trainer needs a better name and documentation
    def __init__(self, ident, model, max_samples, sample_dims):
        self.max_samples = max_samples
        #TODO: Check whether it can be inferred from model.
        self.sample_dims = sample_dims
        super(GentleTrainer, self).__init__(ident, model)

    def minibatch_score(self, f, data):
        batches = iter_minibatches(data, self.max_samples, self.sample_dims, 1)
        score = 0.
        seen_samples = 0.
        for batch in batches:
            this_samples = batch[0].shape[self.sample_dims[0]]
            score += f(*batch) * this_samples
            seen_samples += this_samples
        return ma.scalar(score / seen_samples)

    def handle_update(self, fit_data, eval_data):
        update_losses = {}
        for key, data in eval_data.items():
            update_losses['%s_loss' % key] = self.minibatch_score(
            self.model.score, data)
        return update_losses



class CheckpointTrainer(Trainer):

    def __init__(self, ident, model):
        signal.signal(signal.SIGINT, self._ctrl_c_handler)
        self.stop_next = False
        super(CheckpointTrainer, self).__init__(ident, model)

    def stop(self, stop_info):
        return stop_info or self.stop_next

    def _ctrl_c_handler(self, signal, frame):
        self.stop_next = True

    def save_state(self, filename):
        #Maybe save model?
        if self.current_info:
            return {
                'model_params': self.model.parameters.data.copy(),
                     'info': self.current_info
            }
        return None

    def load_state(self, state):
        assert state is not None
        self.model.parameters.data = state['model_params']
        self.current_info = state['info']


class GentleCheckpointTrainer(CheckpointTrainer, GentleTrainer):

    def __init__(self, ident, model):
        super(GentleCheckpointTrainer, self).__init__(ident, model)