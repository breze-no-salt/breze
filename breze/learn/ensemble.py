def bag(x, z, n_samples):
    idxs = [random.randint(0, x.shape[1] - 1) for _ in range(n_samples)]
    print x.shape
    return x[:, idxs, :], z[:, idxs, :]


def add_to_ensemble(ensemble, train_func, n_samples, X, Z):
    x, z = bag(X, Z, n_samples)
    vx, vz = bag(X, Z, n_samples)
    model, _ = train_func(x, z, vx, vz)
    ensemble.append(model)


def ensemble_predict(ensemble, X):
    return np.mean([i.predict(X) for i in ensemble], axis=0)
