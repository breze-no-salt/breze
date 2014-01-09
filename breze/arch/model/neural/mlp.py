# -*- coding: utf-8 -*-


import theano.tensor as T

from ..util import ParameterSet, Model, lookup
from ..component import transfer, loss as loss_, layer, corrupt


def parameters(n_inpt, n_hiddens, n_output):
    spec = dict(in_to_hidden=(n_inpt, n_hiddens[0]),
                hidden_to_out=(n_hiddens[-1], n_output),
                hidden_bias_0=n_hiddens[0],
                out_bias=n_output)

    zipped = zip(n_hiddens[:-1], n_hiddens[1:])
    spec['hidden_bias_0'] = n_hiddens[0]
    for i, (inlayer, outlayer) in enumerate(zipped):
        spec['hidden_to_hidden_%i' % i] = (inlayer, outlayer)
        spec['hidden_bias_%i' % (i + 1)] = outlayer

    return spec


def exprs(inpt, target, in_to_hidden, hidden_to_hiddens, hidden_to_out,
          hidden_biases, out_bias,
          hidden_transfers, out_transfer,
          loss=None,
          p_dropout_inpt=False, p_dropout_hiddens=False,
          ):

    if not len(hidden_to_hiddens) + 1 == len(hidden_biases) == len(hidden_transfers):
        print (hidden_to_hiddens)
        print (hidden_biases)
        print (hidden_transfers)
        raise ValueError('n_hiddens and hidden_transfers and hidden_biases '
                         'have to be of the same length')

    weights = [in_to_hidden] + hidden_to_hiddens + [hidden_to_out]
    biases = hidden_biases + [out_bias]
    transfers = hidden_transfers + [out_transfer]

    if not p_dropout_inpt:
        p_dropout_inpt = 0
    if not p_dropout_hiddens:
        p_dropout_hiddens = [0] * len(hidden_biases)

    # We append a 0 because dropout makes no sense at the end.
    p_dropouts = p_dropout_hiddens + [0]

    exprs = {}
    last_output = inpt
    if p_dropout_inpt:
        last_output = corrupt.mask(last_output, p_dropout_inpt)

    for i, (w, b, t, d) in enumerate(zip(weights, biases, transfers, p_dropouts)):
        exprs.update(layer.simple(last_output, w, b, t, d, 'layer-%i-' % i))
        last_output = exprs['layer-%i-output' % i]

    # Tidy up a little; we do not want the last layer to be in the exprs dict
    # twice.
    exprs['output'] = exprs['layer-%i-output' % i]
    exprs['output_in'] = exprs['layer-%i-output_in' % i]
    del exprs['layer-%i-output' % i]
    del exprs['layer-%i-output_in' % i]

    if loss is not None:
        f_loss = lookup(loss, loss_)
        loss_rowwise = f_loss(target, last_output).sum(axis=1)
        loss = loss_rowwise.mean()

        exprs['loss_rowwise'] = loss_rowwise
        exprs['loss'] = loss

    return exprs
