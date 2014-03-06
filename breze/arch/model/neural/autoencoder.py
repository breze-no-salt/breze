# -*- coding: utf-8 -*-

# TODO document
# TODO docstring example

def parameters(n_inpt, n_hiddens, tied_weights=False):
    """Return the parameter specification dictionary for an auto encoder.

    Parameters
    ----------

    n_inpt : integer
        Number of inputs of the model.

    n_hiddens : list of integers
        Each item corresponds to one hidden layer of the auto encoder. If
        ``tied_weights`` is set, this has to be symmetric.

    tied_weights : boolean, optional [default: False]
        Flag indicating whether the auto encoder should use tied weights, i.e.
        the n'th weight matrix equals the transpose of the K-n'th weight
        matrix.

    Returns
    -------

    res : dict
        Dictionary specifying the parameters needed.
    """
    # Validate symmetry of layer sizes in case of tied weights.
    if tied_weights:
        for i in range(len(n_hiddens) / 2):
            if not n_hiddens[i] == n_hiddens[-i]:
                raise ValueError('layer sizes need to be symmetric with tied '
                                 'weights')

    if tied_weights:
        spec = dict(in_to_hidden=(n_inpt, n_hiddens[0]),
                    out_bias=n_inpt)
    else:
        spec = dict(in_to_hidden=(n_inpt, n_hiddens[0]),
                    hidden_to_out=(n_hiddens[-1], n_inpt),
                    out_bias=n_inpt)

    zipped = zip(n_hiddens[:-1], n_hiddens[1:])
    for i, (inlayer, outlayer) in enumerate(zipped):
        if tied_weights and i > len(n_hiddens) / 2:
            break
        spec['hidden_to_hidden_%i' % i] = (inlayer, outlayer)
    for i, j in enumerate(n_hiddens):
        spec['hidden_bias_%i' % i] = j
    return spec
