# -*- coding: utf-8 -*-


import numpy as np


def simulate(f_energy_prime, initial_position, initial_momentum,
             n_steps, step_size):
    """Return the position and momentum by simulating Hamiltonian dynamics
    on the energy landscape for `n_steps` leapfrog steps of size `step_size`
    starting out at `initial_position` with `initial_momentum`."""
    position = initial_position.copy()
    momentum = initial_momentum.copy()

    # Make a half step for momentum.
    momentum -= step_size * f_energy_prime(position) / 2

    for i in range(n_steps):
        # Full step for position.
        position += step_size * momentum

        # Make full step for momentum except at last iteration.
        if i < n_steps - 1:
            momentum -= step_size * f_energy_prime(position)

    # Make half step for momentum at the end.
    momentum -= step_size * f_energy_prime(position) / 2

    # Negate momentum at the end of trajectory to make proposal symmetric.
    momentum = -momentum

    return position, momentum

# Numpy does not support summing over several axes. Thus, this is a shorthand
# for this.
def sum(array, axis):
    if not isinstance(axis, (tuple, list)):
        return array.sum(axis)
    axis = list(axis)
    axis.sort()
    for i, ax in enumerate(axis):
        array = array.sum(ax - i)
    return array


def move(f_energy, f_energy_prime, initial_position, n_steps, step_size,
         sample_dim=0):
    """Return a pair (accept_rate, new) where `new` is a new position of
    particles.

    Hamiltonian dynamics are simulated on the energy landscape given by
    `f_energy` and `f_energy_prime`, starting out at `initial_position`. Take
    `n_steps` of `step_size`.

    Depending on metropolis hastings, the resulting positions are either
    rejected (in which case the corresponding initial position is taken)
    or accepted. `accept_rate` specifies the actual ratio of accepted
    positions.

    `positions` can be an array of arbitrary dimensionality. However, the
    axis discriminating to different particles needs to be specified by
    `sample_dim`.
    """
    initial_momentum = np.random.standard_normal(initial_position.shape)
    proposed_position, proposed_momentum = simulate(
        f_energy_prime, initial_position, initial_momentum, n_steps, step_size)

    # Store all axes which are not the sample_dim in a tuple.
    all_but_sample_dim = range(initial_position.ndim)
    all_but_sample_dim.remove(sample_dim)
    all_but_sample_dim = tuple(all_but_sample_dim)

    initial_energy = f_energy(initial_position)
    initial_kinetic = sum(initial_momentum**2, axis=all_but_sample_dim) / 2

    proposed_energy = f_energy(proposed_position)
    proposed_kinetic = sum(proposed_momentum**2, axis=all_but_sample_dim) / 2

    p_accept = np.exp(
        initial_energy - proposed_energy + initial_kinetic - proposed_kinetic)
    accept = (np.random.random(p_accept.shape) < p_accept)
    shape = [1 for _ in initial_position.shape]
    shape[sample_dim] = accept.shape[0]
    accept.shape = shape
    accept_rate = accept.mean()

    new = accept * proposed_position + (1 - accept) * initial_position
    return accept_rate, new


def sample(f_energy, f_energy_prime, position, n_steps,
           desired_accept=0.9,
           initial_step_size=0.01,
           step_size_grow=1.02, step_size_shrink=0.98,
           step_size_min=1E-4, step_size_max=0.25,
           avg_accept_slowness=0.9,
           logfunc=None,
           sample_dim=0):
    """Return a sample from the distribution given by `f_energy`."""
    avg_accept_rate = None
    step_size = initial_step_size
    while True:
        accept_rate, position = move(
            f_energy, f_energy_prime, position, n_steps, step_size, sample_dim)
        yield position

        # In first iteration, don't use moving average.
        if avg_accept_rate is None:
            avg_accept_rate = accept_rate

        # Adjust step size.
        avg_accept_rate = (avg_accept_slowness * avg_accept_rate
                           + (1 - avg_accept_slowness) * accept_rate)
        if avg_accept_rate > desired_accept:
            step_size *= step_size_grow
        else:
            step_size *= step_size_shrink
        step_size = min(step_size_max, max(step_size_min, step_size))
