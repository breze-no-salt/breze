# -*- coding: utf-8 -*-


from ...component import layer, corrupt

# TODO document
# TODO docstrings
# TODO add skip connections to MLPs


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


def exprs(inpt, in_to_hidden, hidden_to_hiddens, hidden_to_out,
          hidden_biases, out_bias,
          hidden_transfers, out_transfer,
          p_dropout_inpt=False, p_dropout_hiddens=False,
          prefix=''):

    if not len(hidden_to_hiddens) + 1 == len(hidden_biases) == len(hidden_transfers):
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

    exprs = dict(('%s%s' % (prefix, k), v) for k, v in exprs.items())

    return exprs
