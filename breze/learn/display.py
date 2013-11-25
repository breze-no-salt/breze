import math

import numpy as np
import matplotlib.pyplot as plt

colors = 'kcmrbyg' * 5


def scatterplot_matrix(X, C=None, symb='o', alpha=1, fig=None):
    """Return a figure containig a scatter plot matrix.

    This is a useful tool for inspecting multi dimensional data. Each dimension
    will be plotted against each dimension as a scatter plot, arranged into a
    matrix. The diagonal will contain histograms.


    Parameters
    ----------

    X : array_like
        2D array containing the points to plot.

    C : array_like
        Class labels (optional). Each row of ``X`` with the same value in ``C``
        will be given the same color in the plots.

    symb : string
        Symbol to use for plotting. Will be forwarded to ``pylab.plot``.

    alpha : float
        Between 0 and 1. Transparency of the points, where 1 means fully
        opaque.

    fig : matplotlib.pyplot.Figure or None
        Figure to plot into. If None, will be created itself.
    """
    N = X.shape[1]
    fig = plt.figure(figsize=figsize) if fig is None else fig
    n_colors = len(set(C)) if C is not None else 0

    for i in range(N):
        for j in range(N):
            ax = fig.add_subplot(N, N, i * N + j + 1)
            plt.setp(ax, yticklabels=[], xticklabels=[])

            if i == j:
                ax.hist(X[:, i], 25)
            else:
                if C is not None:
                    for k in range(n_colors):
                        select = C == k
                        c = colors[k] + symb
                        ax.plot(X[select, j], X[select, i], c, alpha=alpha)
                else:
                    ax.plot(X[:, j], X[:, i], symb, alpha=alpha)
    return fig


def time_series_filter_plot(filters, n_rows=None, n_cols=None, fig=None):
    """Plot filters for time series data.

    Each filter is plotted into its own axis.


    Parameters
    ----------

    filters : array_like
        The argument ``filters`` is expected to be an array of shape
        ``(n_filters, window_size, n_channels)``. ``n_filters`` is the number of
        filter banks, ``window_size`` is the length of a time window and
        ``n_channels`` is the number of different sensors.

    n_rows : int, optional, default: None
        Number of rows for the plot. If not given, inferred from ``n_cols`` to
        match dimensions. If ``n_cols`` is not given as well, both are taken to
        be roughly the square root of the number of filters.

    n_cols : int, optional, default: None
        Number of rows for the plot. If not given, inferred from ``n_rows`` to
        match dimensions. If ``n_rows`` is not given as well, both are taken to
        be roughly the square root of the number of filters.

    fig : Figure, optional
        Figure to plot the axes into. If not given, a new one is created.


    Returns
    -------

    figure : matplotlib figre
        Figure object to save or plot.
    """

    n_filters, window_size, n_channels = filters.shape

    def ceil_int(f):
        return int(math.ceil(f))

    if n_rows is None and n_rows is None:
        n_cols = n_rows = ceil_int(math.sqrt(n_filters))
    elif n_cols is None:
        n_cols = ceil_int(float(n_rows) / n_filters)
    elif n_rows is None:
        n_rows = ceil_int(float(n_cols) / n_filters)

    fig = plt.figure() if fig is None else fig

    axis_num = 1
    for row in range(n_rows):
        for col in range(n_cols):
            ax = plt.subplot(n_rows, n_cols, axis_num)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.plot(filters[axis_num - 1])
            axis_num += 1
            if axis_num > filters.shape[0]:
                break
        if axis_num > filters.shape[0]:
            break
    return fig


# From the matpplotlib cookbook at
# http://www.scipy.org/Cookbook/Matplotlib/HintonDiagrams.
# Adapte to work on an axis which can be given as an argument.

def _blob(ax, x, y, area, colour):
    """Draws a square-shaped blob with the given area (< 1) at
    the given coordinates."""
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    ax.fill(xcorners, ycorners, colour, edgecolor=colour)


def hinton(ax, W, max_weight=None):
    """
    Draws a Hinton diagram  for the matrix `W` to axis `ax`.
    """
    reenable = False

    height, width = W.shape
    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    ax.fill(np.array([0,width,width,0]),np.array([0,0,height,height]),'gray')
    ax.axis('off')
    ax.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y,x]
            if w > 0:
                _blob(ax, _x - 0.5, height - _y + 0.5, min(1,w/max_weight),'white')
            elif w < 0:
                _blob(ax, _x - 0.5, height - _y + 0.5, min(1,-w/max_weight),'black')
