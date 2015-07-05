# -*- coding: utf-8 -*-

import theano.tensor as T
import numpy as np

from distributions import DiagGauss, NormalGauss

from breze.arch.component.misc import inter_gauss_kl

def gauss_normalgauss_kl(p,q):
    n_latent = p.output.shape[1] // 2
    mean = p.output[:, :n_latent]
    var = p.output[:, n_latent:]
    kl = inter_gauss_kl(mean, var, 1e-4)

    return kl

def gauss_gauss_kl(p,q):
    n_latent = p.output.shape[1] // 2
    mean1 = p.output[:, :n_latent]
    var1 = p.output[:, n_latent:]
    mean2 = q.output[:, :n_latent]
    var2 = q.output[:, n_latent:]
    kl = inter_gauss_kl(mean1, var1, mean2, var2)

    return kl

kl_table = { (DiagGauss,NormalGauss): gauss_normalgauss_kl, (DiagGauss,DiagGauss): gauss_gauss_kl }

def kl_div(p,q,sample=False):
	if not sample:
		for i in kl_table:
			if isinstance(p,i[0]) and isinstance(q,i[1]):
				return kl_table[i](p,q)
	else:
		# TODO: implement kl through sampling
		raise NotImplemented()
