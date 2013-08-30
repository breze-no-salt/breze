# -*- coding: utf-8 -*-

import cPickle
import itertools
import gzip

import Image as pil
import numpy as np
import theano
import theano.tensor as T
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from math import sqrt
from math import exp
from matplotlib import widgets
import sys

from climin import Lbfgs
from climin.util import draw_mini_slices

from breze.model.feature import RestrictedBoltzmannMachine as RBM
from breze.util import WarnNaNMode

from utils import tile_raster_images, one_hot

