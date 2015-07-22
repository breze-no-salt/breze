# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T

from breze.arch.construct import neural
from breze.learn.utils import theano_floatx

from base import GenericVariationalAutoEncoder


class ConvolutionalVAE(GenericVariationalAutoEncoder):

    def __init__(self, image_height, image_width, n_channel,
                 recog_n_hiddens_conv,
                 recog_filter_shapes, recog_pool_shapes,
                 recog_n_hiddens_full,
                 recog_transfers_conv, recog_transfers_full,
                 n_latent,
                 gen_n_hiddens_full,
                 gen_n_hiddens_conv,
                 gen_filter_shapes, gen_unpool_factors,
                 gen_transfers_conv, gen_transfers_full,
                 assumptions,
                 recog_strides=None,
                 use_imp_weight=False,
                 batch_size=None,
                 optimizer='adam',
                 max_iter=1000, verbose=False):
        self.image_height = image_height
        self.image_width = image_width
        self.n_channel = n_channel

        self.recog_n_hiddens_conv = recog_n_hiddens_conv
        self.recog_filter_shapes = recog_filter_shapes
        self.recog_pool_shapes = recog_pool_shapes
        self.recog_n_hiddens_full = recog_n_hiddens_full
        self.recog_transfers_conv = recog_transfers_conv
        self.recog_transfers_full = recog_transfers_full
        self.recog_strides = recog_strides

        self.n_latent = n_latent
        self.gen_n_hiddens_conv = gen_n_hiddens_conv
        self.gen_filter_shapes = gen_filter_shapes
        self.gen_unpool_factors = gen_unpool_factors
        self.gen_n_hiddens_full = gen_n_hiddens_full
        self.gen_transfers_conv = gen_transfers_conv
        self.gen_transfers_full = gen_transfers_full

        rec_class = lambda inpt, declare: neural.Lenet(
            inpt, self.image_height, self.image_width, self.n_channel,
            self.recog_n_hiddens_conv,
            self.recog_filter_shapes, self.recog_pool_shapes,
            self.recog_n_hiddens_full,
            self.recog_transfers_conv, self.recog_transfers_full,
            assumptions.latent_layer_size(self.n_latent),
            assumptions.statify_latent,
            strides=self.recog_strides,
            declare=declare)

        gen_class = lambda inpt, declare: neural.DeconvNet2d(
            inpt=inpt, n_inpt=n_latent,
            n_hiddens_full=self.gen_n_hiddens_full,
            n_interim_channel=1,
            n_hiddens_conv=self.gen_n_hiddens_conv,
            filter_shapes=self.gen_filter_shapes,
            unpool_factors=self.gen_unpool_factors,
            hidden_transfers_full=self.gen_transfers_full,
            hidden_transfers_conv=self.gen_transfers_conv,
            output_height=self.image_height,
            output_width=self.image_width,
            n_output_channel=self.n_channel,
            out_transfer_conv=assumptions.statify_visible,
            out_transfer_full='identity',
            declare=declare)

        # TODO n_inpt is not a reasonable input for a convnet; this is why None
        # is used here instead.
        GenericVariationalAutoEncoder.__init__(
            self, None, n_latent,
            assumptions, rec_class, gen_class, use_imp_weight=use_imp_weight,
            batch_size=batch_size, optimizer=optimizer,
            max_iter=verbose, verbose=verbose)

    def _make_start_exprs(self):
        inpt = T.tensor4('inpt')
        inpt.tag.test_value, = theano_floatx(np.ones(
            (3, self.n_channel, self.image_height, self.image_width)))

        if self.use_imp_weight:
            imp_weight = T.tensor4('imp_weight')
            imp_weight.tag.test_value, = theano_floatx(np.ones(
                (3, self.n_channel, self.image_height, self.image_width)))
        else:
            imp_weight = None

        return inpt, imp_weight
