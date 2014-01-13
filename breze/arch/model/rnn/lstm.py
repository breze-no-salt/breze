# -*- coding: utf-8 -*-


import theano
import theano.tensor as T

from ...util import lookup
from ...component import transfer

from rnn import feedforward_layer, leaky_integration
from pooling import pooling_layer

def parameters(n_inpt, n_hiddens, n_output):
    spec = {
        'in_to_hidden': (n_inpt, 4 * n_hiddens[0]),
        'hidden_to_out': (n_hiddens[-1], n_output),
        'hidden_bias_0': 4 * n_hiddens[0],
        'recurrent_0': (n_hiddens[0], 4 * n_hiddens[0]),
        'out_bias': n_output,
        'ingate_peephole_0': (n_hiddens[0],),
        'outgate_peephole_0': (n_hiddens[0],),
        'forgetgate_peephole_0': (n_hiddens[0],)
    }

    zipped = zip(n_hiddens[:-1], n_hiddens[1:])
    for i, (inlayer, outlayer) in enumerate(zipped):
        spec.update({
            'hidden_bias_%i' % (i + 1): 4 * outlayer,
            'hidden_to_hidden_%i': (inlayer, 4 * outlayer),
            'recurrent_%i': (outlayer, 4 * outlayer),
            'ingate_peephole_%i' % (i + 1): (outlayer,),
            'outgate_peephole_%i' % (i + 1): (outlayer,),
            'forgetgate_peephole_%i' % (i + 1): (outlayer,)
        })
    return spec


def lstm_layer(hidden_inpt, hidden_to_hidden,
               ingate_peephole, outgate_peephole, forgetgate_peephole,
               f):
    n_hidden_out = hidden_to_hidden.shape[0]

    def lstm_step(x_t, s_tm1, h_tm1):
        x_t += T.dot(h_tm1, hidden_to_hidden)

        inpt = T.tanh(x_t[:, :n_hidden_out])
        gates = x_t[:, n_hidden_out:]
        inpeep = s_tm1 * ingate_peephole
        outpeep = s_tm1 * outgate_peephole
        forgetpeep = s_tm1 * forgetgate_peephole

        ingate = f(gates[:, :n_hidden_out] + inpeep)
        forgetgate = f(
            gates[:, n_hidden_out:2 * n_hidden_out] + forgetpeep)
        outgate = f(gates[:, 2 * n_hidden_out:] + outpeep)

        s_t = inpt * ingate + s_tm1 * forgetgate
        h_t = f(s_t) * outgate
        return [s_t, h_t]

    (states, hidden_rec), _ = theano.scan(
        lstm_step,
        sequences=hidden_inpt,
        outputs_info=[T.zeros_like(hidden_inpt[0, :, 0:n_hidden_out]),
                      T.zeros_like(hidden_inpt[0, :, 0:n_hidden_out])
                      ])

    return states, hidden_rec


def exprs(inpt, in_to_hidden, hidden_to_hiddens, hidden_to_out,
          hidden_biases, recurrents, out_bias,
          ingate_peepholes, outgate_peepholes, forgetgate_peepholes,
          hidden_transfers, out_transfer, pooling, leaky_coeffs=None):
    exprs = {}

    f_hiddens = [lookup(i, transfer) for i in hidden_transfers]
    f_output = lookup(out_transfer, transfer)

    # First ordinary feedforward layer.
    hidden_in = feedforward_layer(inpt, in_to_hidden, hidden_biases[0])

    # First recurrent layer.
    state, hidden_rec = lstm_layer(
        hidden_in, recurrents[0],
        ingate_peepholes[0], outgate_peepholes[0], forgetgate_peepholes[0],
        f_hiddens[0])

    exprs['state_0'] = state
    exprs['hidden_0'] = hidden_rec

    if leaky_coeffs is not None:
        hidden_rec = leaky_integration(hidden_rec, leaky_coeffs[0])

    exprs['hidden_0'] = hidden_rec

    # Optional further recurrent layers.
    zipped = zip(hidden_to_hiddens, hidden_biases[1:], recurrents[1:],
                 ingate_peepholes[1:], outgate_peepholes[1:],
                 forgetgate_peepholes[1:],
                 f_hiddens[1:])
    for i, (w, b, r, ig, og, fg, t) in enumerate(zipped):
        hidden_m1 = hidden_rec
        hidden_in = feedforward_layer(hidden_m1, w, b)

        state, hidden_rec = lstm_layer(hidden_in, r, ig, og, fg, t)

        if leaky_coeffs is not None:
            hidden_rec = leaky_integration(hidden_rec, leaky_coeffs[i])

        exprs['state_%i' % (i + 1)] = state
        exprs['hidden_%i' % (i + 1)] = hidden_rec

    unpooled = feedforward_layer(hidden_rec, hidden_to_out, out_bias)

    if pooling is None:
        output_in = unpooled
    else:
        output_in = pooling_layer(unpooled, pooling)

    output = f_output(output_in)

    exprs.update(
        {'inpt': inpt,
         'unpooled': unpooled,
         'output_in': output_in,
         'output': output})

    return exprs
