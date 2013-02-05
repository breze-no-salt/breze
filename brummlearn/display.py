import matplotlib.pyplot as plt

colors = 'kcmrbyg' * 5


def scatterplot(X, C=None, symb='o', alpha=1, figsize=(16, 9)):
    N = X.shape[1]
    fig = plt.figure(figsize=figsize)
    n_colors = len(set(C))

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
