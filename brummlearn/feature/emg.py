"""Module that holds various preprocessing routines for emg signals."""

import numpy as np


def integrated(X):
    """Return the sum of the absolute values of a signal.

    :param X: An (t, n, d) array where t is the number of time steps,
        n is the number of different signals and d is the number of
        channels.
    :returns: An (n, d) array."""
    return abs(X).sum(axis=0)


def mean_absolute_value(X):
    """Return the mean absolute value of the signal.

    :param X: An (t, n, d) array where t is the number of time steps,
        n is the number of different signals and d is the number of
        channels.
    :returns: An (n, d) array."""
    return abs(X).mean(axis=0)


def modified_mean_absolute_value_1(X):
    """Return a weighted version of the mean absolute value.

    Instead of equal weight, the first and last quarter of the signal
    are only weighed half."""
    t = X.shape[0]
    weights = np.ones(t)
    weights[:t / 4] *= 0.5
    weights[3 * t / 4:] *= 0.5
    weights = weights[:, np.newaxis, np.newaxis]
    return (abs(X) * weights).mean(axis=0)


def modified_mean_absolute_value_2(X):
    """Return a weighted version of the mean absolute value.

    The central half of the signal has weight one. The beginning and
    the last quarter increase/decrease their weight towards that.

    :param X: An (t, n, d) array where t is the number of time steps,
        n is the number of different signals and d is the number of
        channels.
    :returns: An (n, d) array."""
    t = X.shape[0]
    weights = np.ones(t)
    weights[:t / 4] = np.arange(0, t / 4, 4. / t)
    weights[3 * t / 4:] = np.arange(t / 4, 0, -4. / t) - 4. / t
    weights = weights[:, np.newaxis, np.newaxis]
    return (abs(X) * weights).mean(axis=0)


def mean_absolute_value_slope(X):
    """Return the first derivative of the mean absolute value.

    :param X: An (t, n, d) array where t is the number of time steps,
        n is the number of different signals and d is the number of
        channels.
    :returns: An (n, d) array."""
    mav1 = mean_absolute_value(X[:-1])
    mav2 = mean_absolute_value(X[1:])
    return mav2 - mav1


def variance(X):
    """Return the variance of the signals.

    :param X: An (t, n, d) array where t is the number of time steps,
        n is the number of different signals and d is the number of
        channels.
    :returns: An (n, d) array."""
    return X.std(axis=0)**2


def root_mean_square(X):
    """Return the root mean square of the signals.

    :param X: An (t, n, d) array where t is the number of time steps,
        n is the number of different signals and d is the number of
        channels.
    :returns: An (n, d) array."""
    return np.sqrt((X**2).mean(axis=0))


def waveform_length(X):
    """Return the cumulative length of the waveform over the time segment.
    It is the sum of the absolute values of the first derivative of the signal.

    :param X: An (t, n, d) array where t is the number of time steps,
        n is the number of different signals and d is the number of
        channels.
    :returns: An (n, d) array."""
    return abs(X[1:] - X[:-1]).sum(axis=0)


def zero_crossing(X, threshold=1E-8):
    """Return the amount of times the signal crosses the zero y-axis.

    :param X: An (t, n, d) array where t is the number of time steps,
        n is the number of different signals and d is the number of
        channels.
    :param threshold: Changes below this value are ignored. Useful to surpress
        noise.
    :returns: An (n, d) array."""
    crossings = (np.sign(X[1:]) != np.sign(X[:-1]) &
                 abs(X[1:] - X[:-1]) > threshold)
    n_crossings = crossings.sum(axis=0)
    return n_crossings


def slope_sign_change(X, threshold=1E-8):
    """Return the amount of times the signal changes slope.

    :param X: An (t, n, d) array where t is the number of time steps,
        n is the number of different signals and d is the number of
        channels.
    :param threshold: Changes below this value are ignored. Useful to surpress
        noise.
    :returns: An (n, d) array."""
    deriv = X[1:] - X[:-1]
    return zero_crossing(deriv, threshold)


def willison_amplitude(X, threshold=1E-8):
    """Return the amount of times the difference between two adjacent
    emg segments exceeds a threshold.

    :param X: An (t, n, d) array where t is the number of time steps,
        n is the number of different signals and d is the number of
        channels.
    :param threshold: Changes below this value are ignored. Useful to surpress
        noise.
    :returns: An (n, d) array."""
    return (abs((X[1:] - X[:-1])) > threshold).sum(axis=0)
