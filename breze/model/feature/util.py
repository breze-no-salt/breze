# -*- coding: utf-8 -*-


import numpy as np

from breze.model.neural import MultiLayerPerceptron
from breze.util import lookup_some_key


def feature_func_from_model(model):
    """Return a function which calculates the features of `model` given the
    inputs."""
    feature = model.exprs.get('feature', model.exprs.get('hidden', None))
    if feature is None:
        raise ValueError('could not find feature expression')
    return model.function(['inpt'], feature)


def cascade_layers(n_inpt, layer_classes, layer_kwargs):
    """Return a list of layers where the output size and input size of
    consecutive layers match."""
    layer_in_sizes = [n_inpt] + [i.get('n_feature', i.get('n_hidden')) 
                                 for i in layer_kwargs[:-1]]
    zipped = zip(layer_in_sizes, layer_classes, layer_kwargs)

    layers = []
    for n_inpt, klass, kwargs in zipped:
        kwargs = kwargs.copy()
        kwargs['n_inpt'] = n_inpt
        layers.append(klass(**kwargs))
        layers[-1].parameters.data[:] = np.random.normal(
            0, 0.01, layers[-1].parameters.data.shape)

    return layers


def get_affine_parameters(model):
    """Return the affine parameters of `model`, (weights, bias)."""
    possible_weight_names = 'in_to_hidden', 'in_to_feature', 'in_to_out'
    possible_bias_names = 'hidden_bias', 'out_bias', 'bias'

    weights = lookup_some_key(possible_weight_names, model.parameters)
    bias = lookup_some_key(possible_bias_names, model.parameters, 
                           np.zeros(weights.shape[1]))

    return weights, bias


def mlp_from_cascade(layers, loss):
    """From a list of layers, build up an mlp with the given loss."""
    # Determine transfer functions.
    transfers = [l.feature_transfer for l in layers[:-1]]
    out_transfer = layers[-1].out_transfer

    # Determine sizes.
    sizes = [l.n_feature for l in layers[:-1]]
    n_inpt = layers[0].n_inpt
    n_output = layers[-1].n_output

    # Build up mlp.
    mlp = MultiLayerPerceptron(n_inpt, sizes, n_output, transfers, out_transfer,
                               loss)
    mlp.parameters.data[:] = np.zeros(mlp.parameters.data.shape)

    # Transfer weights and biases.

    weights, biases = [], []
    for l in layers:
        w, b = get_affine_parameters(l)
        weights.append(w)
        biases.append(b)

    weight_names = (
        ['in_to_hidden'] +
        ['hidden_to_hidden_%i' % i for i in range(len(weights) - 2)] +
        ['hidden_to_out'])
    bias_names = ['hidden_bias_%i' % i 
                  for i in range(len(biases) - 1)] + ['out_bias']

    for wname, bname, w, b in zip(weight_names, bias_names, weights, biases):
        mlp.parameters[wname][:] = w
        mlp.parameters[bname][:] = b

    return mlp


