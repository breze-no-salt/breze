# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.gradient import disconnected_grad

from breze.arch.construct.layer.distributions import (
    DiagGauss, NormalGauss,
    assert_no_time, recover_time)
from breze.arch.construct.layer.kldivergence import kl_div
from breze.arch.construct.neural import distributions as neural_dists
from breze.arch.construct.neural import Mlp

from breze.learn.utils import theano_floatx
from breze.arch.util import ParameterSet
from breze.learn.base import (
    UnsupervisedModel, TransformBrezeWrapperMixin,
    ReconstructBrezeWrapperMixin)
from breze.arch.construct.simple import AffineNonlinear

class LocallyLinearTransition(object):

    def _init_transition(self):
        # locally linear transition mlp
        self.A_size = self.n_control * self.n_state
        self.b_size = self.n_state * self.n_state
        self.c_size = self.n_state * self.n_state
        out_size = self.A_size + self.b_size + self.c_size

        self.transition_inpt = T.matrix()
        self.transition_mlp = AffineNonlinear(
            self.transition_inpt, self.n_state,
            self.n_alpha, 'softmax',
            declare=self.parameters.declare)

        self.transition_base_mean = self.parameters.declare(
            (self.n_alpha,
            self.A_size + self.b_size + self.c_size))
        self.transition_base_std = self.parameters.declare(
            (self.n_alpha,
            self.A_size + self.b_size + self.c_size))
        self.transition_base_var = (self.transition_base_std) ** 2 + 1e-5

        # the zeroth transition, turning w_0 into z_0
        if self.zeroth_transition:
            self.zero_transition_inpt = T.matrix()
            self.zero_transition_mlp = Mlp(
                self.zero_transition_inpt, self.n_state,
                [128], self.n_state,
                ['rectifier'], 'identity',
                declare=self.parameters.declare)

    def transition(self, z, u, w, transition_base_epsilon):
        alpha = theano.clone(self.transition_mlp.output,
                             {self.transition_inpt: z})
        if self.uncertain_weights:
            transition_base_sample = DiagGauss(self.transition_base_mean,
                                               self.transition_base_var).sample(
                                               epsilon=transition_base_epsilon)
        else:
            transition_base_sample = self.transition_base_mean

        AsBsCs = T.dot( alpha, transition_base_sample)

        As = AsBsCs[:, :self.A_size].reshape(
        (z.shape[0], self.n_control, self.n_state))
        Bs = AsBsCs[:, self.A_size:self.A_size + self.b_size].reshape(
        (z.shape[0], self.n_state, self.n_state))
        Cs = AsBsCs[:, self.A_size + self.b_size:self.A_size + self.b_size + self.c_size]
        Cs = Cs.reshape((z.shape[0], self.n_state, self.n_state))

        Asu = (u[:, :, np.newaxis] * As).sum(1)
        Bsz = (z[:, :, np.newaxis] * Bs).sum(1)
        Csw = (w[:, :, np.newaxis] * Cs).sum(1)

        return  z + Asu + Bsz + Csw

    def zero_transition(self, w):
        if self.zeroth_transition:
            return theano.clone(self.zero_transition_mlp.output,
                                {self.zero_transition_inpt: w})
        else:
            return w


class NonLinearTransition(object):

    def _init_transition(self):
        # locally linear transition mlp
        self.A_size = self.n_control * self.n_state
        self.b_size = self.n_state * self.n_state
        self.c_size = self.n_state * self.n_state
        out_size = self.A_size + self.b_size + self.c_size
        self.transition_inpt = T.matrix()
        self.transition_mlp = Mlp(
            self.transition_inpt, self.n_state + self.n_control,
            self.n_hiddens_transition, self.n_state,
            self.transfers_transition, 'identity',
            declare=self.parameters.declare)

        self.transition_base_mean = self.parameters.declare(
            (self.n_alpha,
             self.A_size + self.b_size + self.c_size))
        self.transition_base_std = self.parameters.declare(
            (self.n_alpha,
             self.A_size + self.b_size + self.c_size))
        self.transition_base_var = (self.transition_base_std) ** 2 + 1e-5

        # the zeroth transition, turning w_0 into z_0
        if self.zeroth_transition:
            self.zero_transition_inpt = T.matrix()
            self.zero_transition_mlp = Mlp(
                self.zero_transition_inpt, self.n_state,
                [128], self.n_state,
                ['rectifier'], 'identity',
                declare=self.parameters.declare)

    def transition(self, z, u, w, transition_base_epsilon):
        zprime = theano.clone(self.transition_mlp.output,
            {self.transition_inpt: T.concatenate((z, u), 1)})

        return z + zprime + 0.1 * w

    def zero_transition(self, w):
        if self.zeroth_transition:
            return theano.clone(self.zero_transition_mlp.output,
                                {self.zero_transition_inpt: w})
        else:
            return w


class MlpGaussObservationModel(object):

    def _init_observation(self):
        self.observation_inpt = T.matrix()
        self.observation_mlp = neural_dists.MlpDiagConstVarGauss(
            self.observation_inpt, self.n_state,
            self.n_hiddens_gen,
            self.n_obs,
            self.transfers_gen,
            declare=self.parameters.declare)

    def observation(self, z):
        return (theano.clone(self.observation_mlp.mean,
                             {self.observation_inpt: z}),
                theano.clone(self.observation_mlp.var,
                             {self.observation_inpt: z}))


class MlpGaussInitial(object):

    def _init_initial(self):
        self.initial_inpt = T.matrix()
        self.initial_mlp = neural_dists.MlpDiagGauss(
            self.initial_inpt, self.n_obs * self.n_initialobs,
            self.n_hiddens_gen,
            self.n_state,
            self.transfers_gen,
            declare=self.parameters.declare)

    def initial(self, x, epsilon):
        return (theano.clone(self.initial_mlp.sample(epsilon=epsilon),
                             {self.initial_inpt: x.dimshuffle(1, 0, 2).reshape(
                             (-1, self.n_obs * self.n_initialobs))}),
                theano.clone(self.initial_mlp.mean,
                             {self.initial_inpt: x.dimshuffle(1, 0, 2).reshape(
                             (-1, self.n_obs * self.n_initialobs))}),
                theano.clone(self.initial_mlp.var,
                             {self.initial_inpt: x.dimshuffle(1, 0, 2).reshape(
                             (-1, self.n_obs * self.n_initialobs))}))


class RnnGaussInitial(object):

    def _init_initial(self):
        self.initial_inpt = T.tensor3()
        self.initial_rnn = neural_dists.FastDropoutBiRnnDiagGauss(
            self.initial_inpt, self.n_obs,
            self.n_hiddens_gen, self.n_state,
            self.transfers_gen,
            declare=self.parameters.declare)
        self.initial_mlp = DiagGauss(
            self.initial_rnn.mean[0],
            self.initial_rnn.var[0])

    def initial(self, x, epsilon):
        return (theano.clone(self.initial_mlp.sample(epsilon=epsilon),
                             {self.initial_inpt: x}),
                theano.clone(self.initial_mlp.mean, {self.initial_inpt: x}),
                theano.clone(self.initial_mlp.var, {self.initial_inpt: x}))


class MlpDiagUpdate(object):

    def _init_update(self):
        self.beta = self.parameters.declare((1,))

        self.update_inpt = T.matrix()
        self.update_mlp = neural_dists.MlpDiagGauss(
            self.update_inpt, self.n_state + self.n_obs + self.n_control,
            self.n_hiddens_recog, self.n_state,
            self.transfers_recog,
            declare=self.parameters.declare)

    def update(self, z, obs, u, epsilon):
        w = theano.clone(self.update_mlp.sample(epsilon=epsilon),
                         {self.update_inpt: T.concatenate((z,obs,u), 1)})
        w_mean = theano.clone(self.update_mlp.mean,
                              {self.update_inpt: T.concatenate((z,obs,u), 1)})
        w_var = theano.clone(self.update_mlp.var,
                             {self.update_inpt: T.concatenate((z,obs,u), 1)})

        return w, w_mean, w_var

class GenericDeepVariationalBayesFilter(UnsupervisedModel,
                       TransformBrezeWrapperMixin,
                       ReconstructBrezeWrapperMixin):

    sample_dim = 1,

    def __init__(self, n_obs, n_control,
                 n_hiddens_recog, n_state, n_alpha,
                 n_hiddens_transition, n_hiddens_gen, transfers_recog,
                 transfers_transition, transfers_gen, n_samples,
                 zeroth_transition=False, uncertain_weights=True,
                 n_initialobs = 15,
                 use_imp_weight=False, batch_size=None, optimizer='adam',
                 max_iter=1000, verbose=False):
        self.n_inpt = n_obs + n_control
        self.n_latent = n_state
        self.n_output = n_obs
        self.n_initialobs = n_initialobs

        self.n_obs = n_obs
        self.n_control = n_control
        self.n_state = n_state
        self.n_hiddens_recog = n_hiddens_recog
        self.n_alpha = n_alpha
        self.n_hiddens_transition = n_hiddens_transition
        self.n_hiddens_gen = n_hiddens_gen
        self.transfers_recog = transfers_recog
        self.transfers_transition = transfers_transition
        self.transfers_gen = transfers_gen
        self.n_samples = n_samples
        self.zeroth_transition = zeroth_transition
        self.uncertain_weights = uncertain_weights

        n_latent = n_state

        self.use_imp_weight = use_imp_weight
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        self._init_exprs()

    def _make_start_exprs(self):
        inpt = T.tensor3('inpt')
        inpt.tag.test_value, = theano_floatx(np.ones((20, 3, self.n_inpt)))

        if self.use_imp_weight:
            imp_weight = T.tensor3('imp_weight')
            imp_weight.tag.test_value, = theano_floatx(np.ones((20, 3, 1)))
        else:
            imp_weight = None

        return inpt, imp_weight

    def _init_exprs(self):
        inpt, self.imp_weight = self._make_start_exprs()
        obs = inpt[:, :, :self.n_obs]
        us = inpt[:, :, self.n_obs:]
        self.us = us
        self.parameters = ParameterSet()

        n_dim = inpt.ndim
        rng = T.shared_randomstreams.RandomStreams()

        self._init_transition()
        self._init_observation()
        self._init_update()
        self._init_initial()

        def one_step(transition_base_epsilon, epsilon, u, obs,
                     z, w, w_mean, w_var):
            w, w_mean, w_var = self.update(z, obs, u, epsilon)
            z = self.transition(z, u, w, transition_base_epsilon)
            z = theano.gradient.grad_clip(z, -2, 2)
            return z, w, w_mean, w_var

        def gen_one_step(transition_base_epsilon, epsilon, u, z):
            z = self.transition(z, u,
                epsilon * T.sqrt(1.0 / disconnected_grad(self.beta.mean())),
                transition_base_epsilon)
            return z

        self.transition_base_epsilons = rng.normal(size=(us.shape[0],
                                    self.n_alpha,
                                    self.A_size + self.b_size + self.c_size))

        epsilons = rng.normal(size=(us.shape[0],
                                    us.shape[1],
                                    self.n_state))
        self.prior = NormalGauss((us.shape[0],us.shape[1],self.n_state))

        self.weight_prior_mean = self.parameters.declare((1,))
        self.weight_prior_std = self.parameters.declare((1,))
        self.weight_prior_var = (self.weight_prior_std ** 2) + 1e-5
        self.weight_prior = DiagGauss(
            self.weight_prior_mean + T.zeros_like(self.transition_base_mean),
            self.weight_prior_var + T.zeros_like(self.transition_base_var))

        w0, w0_mean, w0_var = self.initial(obs[:self.n_initialobs], epsilons[0])
        z0 = self.zero_transition(w0)
        (z, w, w_mean, w_var), _ = theano.scan(one_step,
            sequences=[self.transition_base_epsilons[1:],
                       epsilons[1:], us[:-1], obs[1:]],
            outputs_info=[z0, T.zeros_like(z0), T.zeros_like(z0),
                          T.zeros_like(z0)])

        self.initial_z = z0
        self.gen_z, _ = theano.scan(gen_one_step,
            sequences=[self.transition_base_epsilons[1:],
                       epsilons[1:], self.us[:-1]],
            outputs_info=[self.initial_z])

        self.z = T.concatenate( ([z0], z), 0)
        self.w = T.concatenate( ([w0], w), 0)
        self.x_mean, self.x_var = self.observation(assert_no_time(self.z))
        self.x_mean = recover_time(self.x_mean, self.z.shape[0])
        self.x_var = recover_time(self.x_var, self.z.shape[0])

        self.gen_z = T.concatenate( ([z0], self.gen_z), 0)
        self.gen_x_mean, self.gen_x_var = self.observation(
                                          assert_no_time(self.gen_z))
        self.gen_x_mean = recover_time(self.gen_x_mean, self.gen_z.shape[0])
        self.gen_x_var = recover_time(self.gen_x_var, self.gen_z.shape[0])

        self.gen = DiagGauss(self.x_mean, self.x_var)

        self.recog_sample = self.w

        if self.use_imp_weight:
            imp_weight = T.addbroadcast(self.imp_weight, n_dim - 1)
        else:
            imp_weight = False

        rec_loss = self.gen.nll(obs)
        self.rec_loss_sample_wise = rec_loss.sum(axis=n_dim - 1)
        self.rec_loss = self.rec_loss_sample_wise.mean()

        output = T.concatenate( (self.x_mean, self.x_var), 2)
        self.gen_output = T.concatenate( (self.gen_x_mean, self.gen_x_var), 2)

        # Create the KL divergence part of the loss.
        # kl_coord_wise gets computed in scan loop
        w_mean = T.concatenate(([w0_mean], w_mean), 0)
        w_var = T.concatenate(([w0_var], w_var), 0)
        self.kl_coord_wise = kl_div(DiagGauss( w_mean, w_var), self.prior,
                                    disconnected_grad(self.beta.mean()))

        if self.use_imp_weight:
            self.kl_coord_wise *= imp_weight
        self.kl_sample_wise = self.kl_coord_wise.sum(axis=n_dim - 1)
        self.kl = self.kl_sample_wise.mean()

        
        if self.uncertain_weights:
            self.weight_kl = kl_div(DiagGauss(self.transition_base_mean,
                                              self.transition_base_var),
                                              self.weight_prior).sum()
            loss = (self.weight_kl / self.n_samples / inpt.shape[0]
                  + self.kl
                  + disconnected_grad(self.beta.mean()) * self.rec_loss)
        else:
            loss = self.kl + disconnected_grad(self.beta.mean()) * self.rec_loss

        self.loss_sample_wise = self.kl_sample_wise + self.rec_loss_sample_wise

        UnsupervisedModel.__init__(self, inpt=inpt,
                                   output=output,
                                   loss=loss,
                                   parameters=self.parameters,
                                   imp_weight=self.imp_weight)

        # TODO: this has to become transform_expr or sth like that
        # TODO: convert distribution parameters to latent stt
        #self.transform_expr_name = self.vae.latent
        self.transform_expr_name = None

    def sample(self, us, gen_steps=None, initial_z=None):
        if getattr(self, 'f_gen_sample', None) is None:
            self.f_gen_sample = self.function([self.us, self.initial_z],
                                              self.gen_output,
                                              on_unused_input='ignore')

        if initial_z == None:
            initial_z = np.random.randn((us.shape[1], self.n_state))

        return self.f_gen_sample(us, initial_z)


class DeepVariationalBayesFilter(
    GenericDeepVariationalBayesFilter,
    LocallyLinearTransition,
    RnnGaussInitial,
    MlpGaussObservationModel,
    MlpDiagUpdate):
    pass
