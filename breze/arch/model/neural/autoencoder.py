# -*- coding: utf-8 -*-


from ...util import lookup
from ...component import loss as loss_, layer, corrupt


def parameters(n_inpt, n_hiddens, tied_weights=False):
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
