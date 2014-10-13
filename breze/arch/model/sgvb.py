# -*- coding: utf-8 -*-


import theano.tensor as T


def exprs(inpt, recog_exprs_func, gen_exprs_func,
          latent_sample,
          latent_key='output', visible_key='output',
          shortcut_key=None):
    """Function returning an expression dictionary.

    Parameters
    ----------

    inpt : Theano variable
        Variable representing the inputs to be modelled.

    recog_exprs_func : callable
        Callable that returns the expression dictionary for the recognition
        model given the input. (I.e. it has one argument.)

    gen_exprs_func : callable
        Callable that returns the expression dictionary for the generating
        model given a sample from the recognition model. (I.e. it has one
        argument.)

    latent_posterior_dist : {'diag_gauss'}
        Identifier of the distribution of the latents given the visibles.

    latent_key : string, optional. Default: 'output'.
        Key to use to retrieve the expression for the latents from the
        expression dictionary returned by ``recog_exprs_func``.

    visible_key : string, optional. Default: 'output'.
        Key to use to retrieve the expression for the reconstructions from the
        expression dictionary returned by ``gen_exprs_func``.

    shortcut_key : string or None, optional. Default: None.
        Key to use to retrieve a "shortcut expression" from the recognition
        model.
        This expression will be fed into the generating network with no sampling
        involved. Useful to add deterministic information from context.

    Examples
    --------

    >>> import theano.tensor as T
    >>> inpt = T.matrix()
    >>> rec_model = lambda x: {'output': T.matrix()}
    >>> gen_model = lambda x: {'output': T.matrix()}
    >>> exprs(inpt, rec_model, gen_model, 'diag_gauss', 'diag_gauss').keys()
    ['sample', 'recog', 'latent', 'gen', 'output']
    """
    recog_exprs = recog_exprs_func(inpt)
    latent = recog_exprs[latent_key]

    rng = T.shared_randomstreams.RandomStreams()
    sample = latent_sample(latent, rng)

    if shortcut_key is None:
        gen_inpt = sample
    else:
        gen_inpt = T.concatenate([sample, recog_exprs[shortcut_key]],
                                 axis=latent.ndim - 1)

    gen_exprs = gen_exprs_func(gen_inpt)
    output = gen_exprs[visible_key]

    res = {
        'inpt': inpt,
        'recog': recog_exprs,
        'gen': gen_exprs,
        'sample': sample,
        'gen_inpt': gen_inpt,
        'latent': latent,
        'output': output,
    }

    if shortcut_key is not None:
        res['shortcut'] = recog_exprs[shortcut_key]

    return res
