# -*- coding: utf-8 -*-

"""Module that contains various functionality for trainers."""

import datetime

from climin import mathadapt as ma
from climin.stops import never, always
from climin.util import clear_info

import score as score_
import report as report_


class Trainer(object):
    """Class representing a Trainer.

    A Trainer object is used to ease bookkeeping of fitting models. This is done
    by composing a trainer out of several basic strategies.

    `Scoring strategy`: The way the score is calculated can be determined by
    the callable stored in the ``_score`` field. For some examples, see the
    module ``breze.learn.trainer.score``.

    `Reporting strategy`: For a report function that is applied to each info
    dictionary during a pause when calling ``.fit()``, ``.report`` can be set.
    For examples, see the module ``breze.learn.trainer.report``.

    `Pause criterion`: When to pause optimization of the model to yield
    control back to the user. Determined by the ``.pause`` field. Contains a
    callable, for example see ``climin.stops``.

    `Interrupt criterion`: When to interrupt optimization of the model to
    yield control back to the user. Determined by the ``.interrupt`` field.
    Contains a callable, for example see ``climin.stops``.

    `Stopping criterion`: When to interrupt optimization of the model to
    yield control back to the user. Determined by the ``.stop`` field.
    Contains a callable, for example see ``climin.stops``.

    Why do we need separate stopping and interrupting criteria? An
    optimization might get interrupted (e.g. by a SIGINT of a shared resource
    system). In order to find out whether the trainer thinks optimization has
    actually finished, the ``.stopped`` field is provided.


    Attributes
    ----------

    model : Model object
        Model that is going to be trained by this trainer.

    _score : callable
        Callable that applies a score function to data. Signature is
        ``f_score, *data``.

    pause : callable
        Callable that given a climin info dictionary determines whether to pause
        fitting.

    stop : callable
        Callable that given a climin info dictionary determines whether to stop
        (i.e. finish) fitting.

    interrupt : callable
        Callable that given a climin info dictionry determines whether to
        interrupt fitting.

    report : callable
        Callable to which the info dictionary of the current optimization is
        passed during each pause.

    best_pars : array_like
        Currently best found parameters according to validation data.

    best_loss : float
        Loss on the validation data of ``best_pars``.

    infos : list of dicts
        List containing all info dictionaries of the estimation.

    current_info : dict
        Last info dictionary.

    eval_data : dictionary
        Dictionary of different data sets for evaluation.

    val_key : string
        Key identifying the data set from ``eval_data`` which is used for
        validation.

    stopped : boolean
        If ``stop`` has returned True once, this is set to True. Otherwise
        False. Useful for distinguishing between interrupt and stop.
    """

    def __init__(self, model, stop, score=score_.simple,
                 pause=always, interrupt=never, report=report_.point_print):
        """Create a Trainer object.

        Parameters
        ----------

        model : Model object
            Model that is going to be trained by this trainer.

        stop : callable
            Callable that given a climin info dictionary determines whether to stop
            (i.e. finish) fitting.

        score : callable, optional
            Callable that applies a score function to data. Signature is
            ``f_score, *data``.

        pause : callable, optional
            Callable that given a climin info dictionary determines whether to pause
            fitting.

        interrupt : callable, optional
            Callable that given a climin info dictionry determines whether to
            interrupt fitting.

        report : callable, optional
            Callable to which the info dictionary of the current optimization is
            passed during each pause.
        """

        self.model = model

        self._score = score
        self.pause = pause
        self.stop = stop
        self.interrupt = interrupt
        self.report = report

        self.best_pars = None
        self.best_loss = float('inf')

        self.infos = []
        self.current_info = None

        self.eval_data = {}
        self.val_key = None
        self.stopped = False

    def score(self, *data):
        return self._score(self.model.score, *data)

    def fit(self, *fit_data):
        """Run ``.iter_fit()`` until it terminates

        Termination will occur when either stop or interrupt is True. During
        each pause, ``.report(info)`` will be executed."""
        for i in self.iter_fit(*fit_data):
            self.report(i)

    def iter_fit(self, *fit_data):
        """Iteratively fit the given training data.

        Generator function containing the main logic of the Trainer object.

        The arguments are of variable length and have to match that of the
        ``model.iter_fit()`` and ultimately the used loss function of that
        model.

        Each iteration of the fitting constitutes of running the optimizer of
        the model until either interrupt or pause returns True.

        In both cases, the generator will yield to the user. Additionally:

            - If interrupt returns True, the generator will stop yielding
            values afterwards.
            - stop will be tested. If it is true it will stop yielding
            afterwards and additionally ``.stopped`` will be set to True
            afterwards.
            - ``best_pars`` and ``best_loss`` will be updated.

        The values yielded from this function will be climin info dictionaries
        stripped from any numpy or gnumpy arrays.
        """
        for info in self.model.iter_fit(*fit_data, info_opt=self.current_info):
            interrupt = self.interrupt(info)
            if self.pause(info) or interrupt:
                info['val_loss'] = ma.scalar(self.score(*self.eval_data[self.val_key]))

                cur_val_loss = info['%s_loss' % self.val_key]
                if cur_val_loss < self.best_loss:
                    self.best_loss = cur_val_loss
                    self.best_pars = self.model.parameters.data.copy()

                info['best_loss'] = self.best_loss
                info['best_pars'] = self.best_pars

                info.update({
                    'datetime': datetime.datetime.now(),
                })

                filtered_info = clear_info(info)

                self.infos.append(filtered_info)
                self.current_info = info
                yield info

                if self.stop(info):
                    self.stopped = True
                    break
                if interrupt:
                    break
