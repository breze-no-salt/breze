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
                  

class rbm_model:

    def __init__(self):
        # Hyperparameters.
        self.n_feature = 100
        self.step_rate = 1E-1
        self.momentum = 0.0
        self.n_gibbs_steps = 1

        # build RBM
        self.rbm = RBM(2, self.n_feature)
        self.f_sample = self.rbm.function(['inpt', 'n_gibbs_steps'], 
                                          ['feature_sample', 'gibbs_sample_visible'])
        self.f_p_feature = self.rbm.function(['inpt'], 'p_feature_given_inpt')
        self.f_free_energy = self.rbm.function(['inpt'], 'free_energy_given_visibles')
        self.init_params()

    def init_params(self):
        self.rbm.parameters.data[:] = np.random.normal(0, 1E-2, 
                                                       self.rbm.parameters.data.shape)
        self.past_iterations = 0

    def sample(self, start, steps):
        _, recons = self.f_sample(start, steps)
        return recons

    def free_energy(self, x):
        return self.f_free_energy(x)

    def train(self, tdata, iterations):
        if len(tdata) == 0:
            return

        in_to_feature_update_m1 = 0
        in_bias_update_m1 = 0
        feature_bias_update_m1 = 0

        for i in range(iterations):
            sys.stdout.write('.')
            in_to_feature_step, in_bias_step, feature_bias_step = self._learn_step(tdata)

            in_to_feature_update = self.momentum * in_to_feature_update_m1 + self.step_rate * in_to_feature_step
            in_bias_update = self.momentum * in_bias_update_m1 + self.step_rate * in_bias_step
            feature_bias_update = self.momentum * feature_bias_update_m1 + self.step_rate * feature_bias_step
    
            self.rbm.parameters['in_to_feature'] += in_to_feature_update
            self.rbm.parameters['in_bias'] += in_bias_step
            self.rbm.parameters['feature_bias'] += feature_bias_step

            in_to_feature_update_m1 = in_to_feature_update
            in_bias_update_m1 = in_bias_update
            feature_bias_update_m1 = feature_bias_update

            self.past_iterations += 1    

    def _learn_step(self, x):
        n = x.shape[0]
        feature_sample, recons = self.f_sample(x, self.n_gibbs_steps)
        recons_features = self.f_p_feature(recons)
        in_to_feature_step = (np.dot(x.T, feature_sample) - np.dot(recons.T, recons_features))
        in_bias_step = (x - recons).mean(axis=0)
        feature_bias_step = (feature_sample - recons_features).mean(axis=0)
        return in_to_feature_step / n, in_bias_step, feature_bias_step




class rbm_plot:
    
    def __init__(self):

        # parameters
        self.res = 0.05
        self.training_iterations = 1000
                       
        # create rbm model                                              
        self.model = rbm_model()

        # points
        self.training = [ ]
        self.sample_point = (0, 0)

        # initialize plot
        fig = plt.figure()
        self.ax = plt.axes()       
        self.h_free_energy = None
        self.h_colorbar = None
        self.h_training_points = None
        self.h_hexbin = None
        self.plot_free_energy()

        # register events
        cid = fig.canvas.mpl_connect('button_press_event', self.ax_onclick)

        ax_train = plt.axes([0.9, 0.0, 0.1, 0.1])
        self.b_train = widgets.Button(ax_train, 'Train')
        self.b_train.on_clicked(self.b_train_onclick)

        ax_clear_trainingset = plt.axes([0.9, 0.1, 0.1, 0.1])
        self.b_clear_trainingset = widgets.Button(ax_clear_trainingset, 'Clear ts.')
        self.b_clear_trainingset.on_clicked(self.b_clear_trainingset_onclick)

        ax_reset_rbm = plt.axes([0.9, 0.2, 0.1, 0.1])
        self.b_reset_rbm = widgets.Button(ax_reset_rbm, 'Reset RBM')
        self.b_reset_rbm.on_clicked(self.b_reset_rbm_onclick)

        ax_sample = plt.axes([0.9, 0.3, 0.1, 0.1])
        self.b_sample = widgets.Button(ax_sample, 'Sample')
        self.b_sample.on_clicked(self.b_sample_onclick)

        ax_histogram = plt.axes([0.9, 0.4, 0.1, 0.1])
        self.b_histogram = widgets.Button(ax_histogram, 'Histrogram')
        self.b_histogram.on_clicked(self.b_histogram_onclick)

    def show(self):
        plt.show()

    def plot_free_energy(self):  
        plt.axes(self.ax)

        xgrid, ygrid = np.meshgrid(np.arange(0, 1, self.res), 
                                   np.arange(0, 1, self.res))
        px = xgrid.reshape(-1)
        py = ygrid.reshape(-1)
        p = np.array([px, py]).T
        fe = self.model.free_energy(p)
        #fe = np.exp(f_free_energy(p))
        fegrid = fe.reshape(xgrid.shape)
        #fegrid = ygrid

        if self.h_free_energy == None:
            self.h_free_energy = plt.imshow(fegrid, cmap=cm.gray, 
                                            origin='lower', extent=(0,1,0,1))
            self.h_colorbar = plt.colorbar()
        else:
            self.h_free_energy.set_data(fegrid)
            self.h_free_energy.autoscale()
        
        plt.title('%d iterations' % self.model.past_iterations)
        plt.draw()

    def plot_points(self):
        plt.axes(self.ax)
        if self.h_training_points != None:
            while len(self.h_training_points) > 0:
                self.h_training_points.pop(0).remove()
        self.h_training_points = plt.plot([t[0] for t in self.training], 
                                          [t[1] for t in self.training],
                                          'rx',
                                          self.sample_point[0],
                                          self.sample_point[1],
                                          'bx')
        plt.draw()

    def plot_sampling_histogram(self):
        plt.axes(self.ax)

        if self.h_hexbin != None:
            self.h_hexbin.remove()
            self.h_hexbin = None
            return

        batch_size = 100
        interval = 10
        iterations = 100

        x = np.array([])
        y = np.array([])
        batch = np.zeros((batch_size, 2))
        for i in range(iterations):
            sys.stdout.write('.')
            batch = self.model.sample(batch, interval)
            x = np.append(x, batch[:,0])
            y = np.append(y, batch[:,1])

        self.h_hexbin = plt.hexbin(x, y, 
                                   cmap=cm.gray, extent=(0,1,0,1))

    # event handlers
    def ax_onclick(self, event):
        if event.inaxes == self.ax:
            if event.button == 1:
                self.training.append((event.xdata, event.ydata))
            elif event.button == 3:
                self.training = [t for t in self.training if sqrt((t[0]-event.xdata)**2 +
                                                                  (t[1]-event.ydata)**2) > 0.02]
            elif event.button == 2:
                self.sample_point = (event.xdata, event.ydata)                                    
            self.plot_points()

    def b_train_onclick(self, event):
        self.model.train(np.array(self.training), self.training_iterations)
        self.plot_free_energy()

    def b_clear_trainingset_onclick(self, event):
        self.training = [ ]
        self.plot_points()

    def b_reset_rbm_onclick(self, event):
        self.model.init_params()
        self.plot_free_energy()

    def b_sample_onclick(self, event):
        s = self.model.sample(np.array([np.array(self.sample_point)]), 1)
        self.sample_point = (s[0,0], s[0,1])
        self.plot_points()

    def b_histogram_onclick(self, event):
        self.plot_sampling_histogram()


# main
if __name__ == '__main__':
    p = rbm_plot()
    p.show()

