import theano

from breze.component import norm, corrupt as _corrupt
from breze.util import lookup


def corrupt(exprs, name, typ, pars):
    f_corrupt = lookup(typ, _corrupt)
    if 'true_loss' not in exprs:
        exprs['true_loss'] = exprs['loss']
    uncorrupted = exprs[name]
    corrupted = f_corrupt(uncorrupted, **pars)
    exprs['loss'] = theano.clone(exprs['loss'], {uncorrupted: corrupted})
