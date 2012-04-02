# -*- coding: utf-8 -*-


import numpy as np
from breze.model.sequential import LinearDynamicalSystem

n_inpt, n_hidden = 2, 2

# Create some data.
t = np.arange(0, 1, 0.001)
trajectory_x = np.sin(t)
trajectory_y = np.cos(t)
observed_x = trajectory_x + np.random.standard_normal(trajectory_x.shape) * 0.1
observed_y = trajectory_y + np.random.standard_normal(trajectory_y.shape) * 0.1
X = np.vstack([observed_x, observed_y]).T[:, np.newaxis]
print X.shape


lds = LinearDynamicalSystem(2, 2)
lds.parameters['transition'][:] = np.eye(2)
lds.parameters['emission'][:] = np.eye(2)
lds.parameters['visible_noise_mean'] = np.zeros(2)
lds.parameters['hidden_noise_mean'] = np.zeros(2)
lds.parameters['hidden_mean_initial'] = np.zeros(2)
lds.parameters['visible_noise_cov'] = np.eye(2) * 0.01
lds.parameters['hidden_noise_cov'] = np.eye(2) * 0.00001
lds.parameters['hidden_cov_initial'] = np.eye(2) * 0.01

smooth = lds.function(['inpt'], 'smoothed_means')
smoothed = smooth(X)

import pylab

pylab.plot(X[:, 0, 0], X[:, 0, 1], 'ro', alpha=.3, label='observed')
pylab.plot(smoothed[:, 0, 0], smoothed[:, 0, 1], 'bo', alpha=.3, label='filtered')
pylab.plot(trajectory_x, trajectory_y, 'go', alpha=.3, label='truth')
pylab.legend()
pylab.show()
