# -*- coding: utf-8 -*-


from ...component import layer, corrupt

# TODO docstrings
# TODO add skip connections to MLPs


def parameters(n_inpt, n_hiddens, n_output):
    """Return the parameter specification dictionary for an MLP.

    Parameters
    ----------

    n_inpt : integer
        Number of inputs of the model.

    n_hiddens : list of integers
        Each item corresponds to one hidden layer of the mlp.

    n_output : integer
        Number of outputs of the model.

    Returns
    -------

    res : dict
        Dictionary specifying the parameters needed.
    """
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
    """Return the expressions for an mlp.

    Parameters
    ----------

    inpt : Theano variable
        Variable of shape ``(n, d)`` representing the input to the model.

    in_to_hidden : Theano variable
        Variable of shape ``(d, h1)`` represeting the input to hidden weight
        matrix.

    hidden_to_hiddens : list of Theano variables
        List of theano variables representing the weight matrices between hidden
        layers. The first dimension of any matrix has to match the second of the
        preceeding matrix.

    hidden_to_out : Theano variable
        Variable of shape ``(hk, o)`` represeting the hidden to output weight
        matrix.

    hidden_biases : List of Theano variables
        Each item represents the hidden bias for the corresponding layer.

    out_bias : Theano variable
        Output bias.

    hidden_transfers : List of strings or callables.
        Each item is a transfer function mapping a Theano matrix to a Theano
        matrix of the same size. If a string, such a function will be looked up
        by that name in ``breze.arch.component.transfer``.

    out_transfer : String or callable.
        transfer function mapping a Theano matrix to a Theano matrix of the same
        size. If a string, such a function will be looked up by that name in
        ``breze.arch.component.transfer``.

    p_dropout_inpt : Theano scalar or float
        Every input will be randomly set to zero with that probability.

    p_dropout_hiddens : List of Theano scalars or floats
        Each item of a layer will be set to zero with the given probability.

    prefix : string, optional [default: '']
        The key of each expression will be prefixed with this string in the
        result dict.


    Returns
    -------

    exprs : Dictionary with expressions of the model.
    """

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
