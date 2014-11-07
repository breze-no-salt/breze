import numpy as np

from sklearn.utils.extmath import randomized_svd


def many_dot(*args):
    r = args[0]
    for a in args[1:]:
        r = np.dot(r, a)
    return r


def make_data_matrix(x):
    d = np.zeros((x.shape[0], x.shape[0] * x.shape[1]))
    for i in range(x.shape[0]):
        if i == 0:
            d[i:, i * x.shape[1]:(i +  1) * x.shape[1]] = x
        else:
            d[i:, i * x.shape[1]:(i +  1) * x.shape[1]] = x[:-i]
    return d


def make_big_xi_r(xs):
    Ds = [make_data_matrix(x) for x in xs]
    max_length = max(D.shape[0] for D in Ds)
    total_length = sum(D.shape[0] for D in Ds)

    big_D = np.zeros((total_length, max_length * xs[0].shape[1]))
    start = 0
    for D in Ds:
        stop = start + D.shape[0]
        big_D[start:stop, :D.shape[1]] = D
        start = stop

    big_r = np.zeros((total_length, total_length))
    n = 0
    for x in xs:
        l = x.shape[0]
        r = np.zeros((l, l))
        r[np.arange(1, l), np.arange(l - 1)] = 1
        big_r[n:n + l, n:n + l] = r
        n += l

    return big_D, big_r


def recursive_pca(xs, n):
    xi, R = make_big_xi_r(xs)

    v, s, u = randomized_svd(xi, n)

    vp = v[:, :n]
    sp = s[:n]
    up = u[:n, :]

    A = up[:, :xs[0].shape[1]]
    sinv = 1. / (sp)

    B = many_dot(np.diag(sp), vp.T, R.T, vp, np.diag(sinv)).T

    return A, B


def encode(seq, A, B):
    code = np.zeros((B.shape[0]))
    for i in range(seq.shape[0]):
        code[...] = np.dot(A, seq[i]) + np.dot(B, code)
    return code


def decode(yT, A, B, T):
    x = np.empty((T, A.shape[1]))
    y = np.empty((T, B.shape[0]))
    y[-1] = yT
    for t in reversed(range(T)):
        x[t] = np.dot(A.T, y[t])
        if t != 0:
            y[t - 1] = np.dot(B.T, y[t])
    return x


class RecursivePca(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, xs):
        self.A, self.B = recursive_pca(xs, self.n_components)

    def transform(self, xs):
        codes = []
        for x in xs:
            codes.append(encode(x, self.A, self.B))
        return codes

    def inv_transform(self, ys, ts):
        reconstructs = []
        for y, t in zip(ys, ts):
            reconstructs.append(decode(y, self.A, self.B, t))
        return reconstructs
