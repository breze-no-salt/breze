# -*- coding: utf-8 -*-


import os
import sys
import h5py

import theano
import breze.learn.sgvb as sgvb
from breze.learn import base
from breze.learn.data import interleave, padzeros, split
from breze.learn.trainer.trainer import Trainer
from breze.learn.trainer.report import OneLinePrinter
import climin.initialize
import numpy as np

from breze.learn.sgvb import storn

class GaussConstVarGaussStorn(storn.StochasticRnn,
                              storn.GaussLatentBiStornMixin,
                              storn.ConstVarGaussVisibleStornMixin):
    pass