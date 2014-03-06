# -*- coding: utf-8 -*-


from ..util import get_named_variables
from ..component import layer


def parameters(n_inpt, n_output):
    """Return the parameter specification for a linear model.

    Parameters
    ----------

    n_inpt : integer
        Number of inputs of the model.

    n_output : integer
        Number of outputs of the model.

    Returns
    -------

    res : dict
        Dictionary containing a map from parameters to their shape.
    """
    return {'in_to_out': (n_inpt, n_output),
            'bias': n_output}


def exprs(inpt, weights, bias, out_transfer):
    """Return the expressions for a linear model.

    Parameters
    ----------

    inpt : Theano variable.
        Theano matrix of shape ``(n, d)`` representing the input to the model.

    weights : Theano variable
        Theano matrix of shape ``(d, e)`` representing the linear transform.

    bias : Theano variable
        Theano vector of shape ``(e,)`` representing the offset.

    out_transfer : function or string
        If function should map a Theano variable to another Theano variable of
        the same shape. If string, should be the name of a function in
        ``breze.arch.component.transfer``.

    Returns
    -------

    exprs : dict
        Containing ``output_in`` and ``output``, which represent the output of
        the model before and after application of ``transfer``.
    """
    exprs = layer.simple(inpt, weights, bias, out_transfer)
    output_in, output = exprs['output_in'], exprs['output']

    return get_named_variables(locals())
