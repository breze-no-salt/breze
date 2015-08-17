# -*- coding: utf-8 -*-


from base import GenericVariationalAutoEncoder

from breze.arch.construct.layer.distributions import NormalGauss
from breze.arch.construct.neural import distributions as neural_dists


class MlpGaussLatentVAEMixin(object):

    def make_prior(self, sample):
        return NormalGauss(sample.shape)

    def make_recog(self, inpt):
        return neural_dists.MlpDiagGauss(
            inpt, self.n_inpt,
            self.n_hiddens_recog,
            self.n_latent,
            self.recog_transfers,
            out_transfer_mean='identity',
            out_transfer_var=lambda x: x ** 2 + 1e-5,
            declare=self.parameters.declare)


class MlpGaussVisibleVAEMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.MlpDiagGauss(
            latent_sample, self.n_latent,
            self.n_hiddens_gen,
            self.n_inpt,
            self.gen_transfers,
            # TODO where to get the transfers from?
            # remove lambda
            out_transfer_mean='identity',
            out_transfer_var=lambda x: x ** 2 + 1e-5,
            declare=self.parameters.declare)


class MlpGaussConstVarVisibleVAEMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.MlpDiagConstVarGauss(
            latent_sample, self.n_latent,
            self.n_hiddens_gen,
            self.n_inpt,
            self.gen_transfers,
            # TODO where to get the transfers from?
            declare=self.parameters.declare)


class MlpBernoulliVisibleVAEMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.MlpBernoulli(
            latent_sample, self.n_latent,
            self.n_hiddens_gen,
            self.n_inpt,
            self.gen_transfers,
            declare=self.parameters.declare)


class FastDropoutMlpGaussLatentVAEMixin(object):

    def make_prior(self, sample):
        return NormalGauss(sample.shape)

    def make_recog(self, inpt):
        return neural_dists.FastDropoutMlpDiagGauss(
            inpt, self.n_inpt,
            self.n_hiddens_recog,
            self.n_latent,
            self.recog_transfers,
            out_transfer='identity',
            dropout_parameterized=True,
            declare=self.parameters.declare)


class FastDropoutMlpGaussVisibleVAEMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.FastDropoutMlpDiagGauss(
            latent_sample, self.n_latent,
            self.n_hiddens_gen,
            self.n_inpt,
            self.gen_transfers,
            out_transfer='identity',
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            declare=self.parameters.declare)


class FastDropoutMlpBernoulliVisibleVAEMixin(object):

    def make_gen(self, latent_sample):
        return neural_dists.FastDropoutMlpBernoulli(
            latent_sample, self.n_latent,
            self.n_hiddens_gen,
            self.n_inpt,
            self.gen_transfers,
            out_transfer='sigmoid',
            p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens,
            declare=self.parameters.declare)


class VariationalAutoEncoder(GenericVariationalAutoEncoder):

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 use_imp_weight=False,
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):
        self.n_hiddens_recog = n_hiddens_recog
        self.n_hiddens_gen = n_hiddens_gen
        self.recog_transfers = recog_transfers
        self.gen_transfers = gen_transfers

        super(VariationalAutoEncoder, self).__init__(
            n_inpt, n_latent,
            use_imp_weight=use_imp_weight,
            batch_size=batch_size, optimizer=optimizer,
            max_iter=verbose, verbose=verbose)


class FastDropoutVariationalAutoEncoder(VariationalAutoEncoder):

    def __init__(self, n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
                 recog_transfers, gen_transfers,
                 p_dropout_inpt=.1, p_dropout_hiddens=.1,
                 use_imp_weight=False,
                 batch_size=None, optimizer='rprop',
                 max_iter=1000, verbose=False):
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens

        super(FastDropoutVariationalAutoEncoder, self).__init__(
            n_inpt, n_hiddens_recog, n_latent, n_hiddens_gen,
            recog_transfers, gen_transfers,
            use_imp_weight=use_imp_weight,
            batch_size=batch_size, optimizer=optimizer,
            max_iter=verbose, verbose=verbose)
