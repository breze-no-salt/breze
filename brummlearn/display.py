import math

import matplotlib.pyplot as plt

colors = 'kcmrbyg' * 5


def scatterplot(X, C=None, symb='o', alpha=1, figsize=(16, 9)):
    N = X.shape[1]
    fig = plt.figure(figsize=figsize)
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

    axisNum = 0
    for row in range(n_rows):
        for col in range(n_cols):
            ax = plt.subplot(n_rows, n_cols, axisNum)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.plot(filters[axisNum])
            axisNum += 1
    return fig
