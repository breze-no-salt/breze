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
                                                                               
# Hyperparameters.
n_feature = 50
step_rate = 1E-1
momentum = 0.0
n_gibbs_steps = 1

# build RBM
rbm = RBM(2, n_feature)
f_sample = rbm.function(['inpt', 'n_gibbs_steps'], 
                        ['feature_sample', 'gibbs_sample_visible'])
f_p_feature = rbm.function(['inpt'], 'p_feature_given_inpt')
f_free_energy = rbm.function(['inpt'], 'free_energy_given_visibles')

def learn_step(x):
    n = x.shape[0]
    feature_sample, recons = f_sample(x, n_gibbs_steps)
    recons_features = f_p_feature(recons)
    in_to_feature_step = (np.dot(x.T, feature_sample) - np.dot(recons.T, recons_features))
    in_bias_step = (x - recons).mean(axis=0)
    feature_bias_step = (feature_sample - recons_features).mean(axis=0)

    return in_to_feature_step / n, in_bias_step, feature_bias_step

def sample_from_rbm(start, steps):
    _, recons = f_sample(start, steps)
    return recons

# initialize plot
res = 0.05
xgrid, ygrid = np.meshgrid(np.arange(0, 1, res), np.arange(0, 1, res))
px = xgrid.reshape(-1)
py = ygrid.reshape(-1)
fig = plt.figure()
ax = plt.axes()

# training points
training = [ ]
sample_point = (0, 0)

def init_params():
    global rbm, past_iterations

    rbm.parameters.data[:] = np.random.normal(0, 1E-2, rbm.parameters.data.shape)
    past_iterations = 0

def train(tdata, iterations):
    global rbm, past_iterations

    if len(tdata) == 0:
        return
    print tdata

    in_to_feature_update_m1 = 0
    in_bias_update_m1 = 0
    feature_bias_update_m1 = 0

    for i in range(iterations):
        sys.stdout.write('.')
        in_to_feature_step, in_bias_step, feature_bias_step = learn_step(tdata)

        in_to_feature_update = momentum * in_to_feature_update_m1 + step_rate * in_to_feature_step
        in_bias_update = momentum * in_bias_update_m1 + step_rate * in_bias_step
        feature_bias_update = momentum * feature_bias_update_m1 + step_rate * feature_bias_step
    
        rbm.parameters['in_to_feature'] += in_to_feature_update
        rbm.parameters['in_bias'] += in_bias_step
        rbm.parameters['feature_bias'] += feature_bias_step

        in_to_feature_update_m1 = in_to_feature_update
        in_bias_update_m1 = in_bias_update
        feature_bias_update_m1 = feature_bias_update

        past_iterations += 1
    
h_free_energy = None
def plot_free_energy():
    global h_free_energy
    
    plt.axes(ax)
    if h_free_energy != None:
        h_free_energy.remove()

    p = np.array([px, py]).T
    #fe = f_free_energy(p)
    fe = np.exp(f_free_energy(p))
    fegrid = fe.reshape(xgrid.shape)
    #fegrid = ygrid

    h_free_energy = plt.imshow(fegrid, cmap=cm.gray, origin='lower', extent=(0,1,0,1))
    #plt.colorbar()
    plt.title('%d iterations' % past_iterations)
    plt.draw()

h_training_points = None
def plot_training_points():
    global h_training_points

    plt.axes(ax)
    if h_training_points != None:
        while len(h_training_points) > 0:
            h_training_points.pop(0).remove()
    h_training_points = plt.plot([t[0] for t in training], 
                                 [t[1] for t in training],
                                 'rx',
                                 sample_point[0],
                                 sample_point[1],
                                 'bx')
    plt.draw()

# event handlers
def ax_onclick(event):
    global training, sample_point
    if event.inaxes == ax:
        if event.button == 1:
            training.append((event.xdata, event.ydata))
        elif event.button == 3:
            training = [t for t in training if sqrt((t[0]-event.xdata)**2 +
                                                    (t[1]-event.ydata)**2) > 0.02]
        elif event.button == 2:
            sample_point = (event.xdata, event.ydata)                                    
        plot_training_points()

def b_train_onclick(event):
    train(np.array(training), 1000)
    plot_free_energy()

def b_clear_trainingset_onclick(event):
    global training
    training = [ ]
    plot_training_points()

def b_reset_rbm_onclick(event):
    init_params()
    plot_free_energy()

def b_sample_onclick(event):
    global sample_point
    s = sample_from_rbm(np.array([np.array(sample_point)]), 1)
    sample_point = (s[0,0], s[0,1])
    plot_training_points()

# register events
cid = fig.canvas.mpl_connect('button_press_event', ax_onclick)
ax_train = plt.axes([0.9, 0.0, 0.1, 0.1])
b_train = widgets.Button(ax_train, 'Train')
b_train.on_clicked(b_train_onclick)
ax_clear_trainingset = plt.axes([0.9, 0.1, 0.1, 0.1])
b_clear_trainingset = widgets.Button(ax_clear_trainingset, 'Clear ts.')
b_clear_trainingset.on_clicked(b_clear_trainingset_onclick)
ax_reset_rbm = plt.axes([0.9, 0.2, 0.1, 0.1])
b_reset_rbm = widgets.Button(ax_reset_rbm, 'Reset RBM')
b_reset_rbm.on_clicked(b_reset_rbm_onclick)
ax_sample = plt.axes([0.9, 0.3, 0.1, 0.1])
b_sample = widgets.Button(ax_sample, 'Sample')
b_sample.on_clicked(b_sample_onclick)

# initial plot
init_params()
plot_free_energy()

plt.show()

