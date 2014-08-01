# -*- coding: utf-8 -*-
import climin.initialize
from breze.learn.mlp import Mlp
import numpy as np



def produce_mlp_results(seed):
    optimizer = ('rmsprop', {'decay': 0.9,
    'momentum': 0.85,
    'step_rate': 0.00005,
    'step_rate_max': 0.05,
    'step_rate_min': 1e-06})


    size_inpt = 10
    size_output = 1
    max_iter = 20
    examples = 15

    random_state = np.random.RandomState(seed)
    random_state_2 = np.random.RandomState(3)

    model = Mlp(size_inpt, [10], size_output, ['rectifier'], 'sigmoid', 'squared',
                optimizer=optimizer, batch_size=1, random_state=random_state)

    climin.initialize.randomize_normal(model.parameters.data, 0, 1, random_state=random_state_2)
    parameters_original = model.parameters.data.copy()

    X = random_state_2.rand(examples*size_inpt).reshape(examples, size_inpt)
    Z = np.ones((examples, size_output))

    for info in model.iter_fit(X, Z):
        if info['n_iter'] == max_iter:
            break

    parameters_final = model.parameters.data.copy()

    return parameters_original, parameters_final


def test_parameter_set_data_change():
    #Try several times and with different seeds
    for i in range(10):
        parameters_original1, parameters_final1 = produce_mlp_results(i+1)
        parameters_original2, parameters_final2 = produce_mlp_results(i+1)
        parameters_original3, parameters_final3 = produce_mlp_results(i+2)
        assert np.allclose(parameters_original1, parameters_original2)
        assert not np.allclose(parameters_original1, parameters_final1)
        assert np.allclose(parameters_final1, parameters_final2)
        assert np.allclose(parameters_original2, parameters_original3)
        assert not np.allclose(parameters_final2, parameters_final3)


test_parameter_set_data_change()