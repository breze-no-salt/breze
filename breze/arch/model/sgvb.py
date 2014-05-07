# -*- coding: utf-8 -*-


import theano.tensor as T


def exprs(inpt, recog_exprs_func, gen_exprs_func, visible_dist,
          latent_posterior_dist,
          latent_key='output', visible_key='output'):
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

    visible_dist : {'diag_gauss', 'bern'}
        Identifier of the distribution of the visibles given the latents.

    latent_posterior_dist : {'diag_gauss'}
        Identifier of the distribution of the latents given the visibles.

    latent_key : string, optional. Default: 'output'.
        Key to use to retrieve the expression for the latents from the
        expression dictionary returned by ``recog_exprs_func``.

    visible_key : string, optional. Default: 'output'.
        Key to use to retrieve the expression for the reconstructions from the
        expression dictionary returned by ``gen_exprs_func``.


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

    if latent_posterior_dist == 'diag_gauss':
        n_latent = recog_exprs[latent_key].shape[1] // 2
        latent_mean = latent[:, :n_latent]
        latent_var = latent[:, n_latent:]
        rng = T.shared_randomstreams.RandomStreams()
        noise = rng.normal(size=latent_mean.shape)
        sample = latent_mean + T.sqrt(latent_var + 1e-8) * noise
    else:
        raise ValueError('unknown latent posterior distribution %s' %
                         latent_posterior_dist)

    gen_exprs = gen_exprs_func(sample)
    output = gen_exprs[visible_key]

    return {
        'recog': recog_exprs,
        'gen': gen_exprs,
        'sample': sample,
        'latent': latent,
        'output': output,
    }
