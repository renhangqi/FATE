#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import operator

import numpy as np

from federatedml.linear_model.caesar_model.caesar_base import CaesarBase
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.param.caesar_param import LogisticRegressionParam
from federatedml.protobuf.generated import lr_model_meta_pb2, lr_model_param_pb2
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.spdz import SPDZ
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.transfer_variable.transfer_class.caesar_model_transfer_variable import CaesarModelTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroLRBase(CaesarBase):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroLogisticRegression'
        self.model_param_name = 'HeteroLogisticRegressionParam'
        self.model_meta_name = 'HeteroLogisticRegressionMeta'
        self.mode = consts.HETERO
        self.cipher = None
        self.batch_generator = None
        self.gradient_loss_operator = None
        self.converge_procedure = None
        self.model_param = LogisticRegressionParam()
        self.features = None
        self.labels = None

    def _init_model(self, params: LogisticRegressionParam):
        super()._init_model(params)
        self.cipher = PaillierEncrypt()
        self.cipher.generate_key(self.model_param.encrypt_param.key_length)
        self.transfer_variable = CaesarModelTransferVariable()
        # if self.role == consts.GUEST:
        #     self.batch_generator.register_batch_generator(self.transfer_variable, has_arbiter=False)
        # else:
        #     self.batch_generator.register_batch_generator(self.transfer_variable)

    def get_model_summary(self):
        # TODO
        return

    def share_init_model(self, w, fix_point_encoder, n_iter=-1):
        source = [w, self.other_party]
        if self.local_party.role == consts.GUEST:
            wb, wa = (
                fixedpoint_numpy.FixedPointTensor.from_source(f"wb_{n_iter}", source[0],
                                                              encoder=fix_point_encoder,
                                                              q_field=self.random_field),
                fixedpoint_numpy.FixedPointTensor.from_source(f"wa_{n_iter}", source[1],
                                                              encoder=fix_point_encoder,
                                                              q_field=self.random_field),
            )
            return wb, wa
        else:
            wa, wb = (
                fixedpoint_numpy.FixedPointTensor.from_source(f"wa_{n_iter}", source[0],
                                                              encoder=fix_point_encoder,
                                                              q_field=self.random_field),
                fixedpoint_numpy.FixedPointTensor.from_source(f"wb_{n_iter}", source[1],
                                                              encoder=fix_point_encoder,
                                                              q_field=self.random_field),
            )
            return wa, wb

    def cal_prediction(self, w_self, w_remote, features, spdz, suffix):
        raise NotImplementedError("Should not call here")

    def compute_gradient(self, wa, wb, error, suffix):
        raise NotImplementedError("Should not call here")

    def transfer_pubkey(self):
        raise NotImplementedError("Should not call here")

    def fit(self, data_instances, validate_data=None):
        self.header = data_instances.schema["header"]
        self.fit_binary(data_instances, validate_data)

    def fit_binary(self, data_instances, validate_data=None):
        LOGGER.info("Start to caesar hetero_lr")

        self.validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        # self.batch_generator.initialize_batch_generator(data_instances, self.batch_size)
        model_shape = self.get_features_shape(data_instances)
        w = self.initializer.init_model(model_shape, init_params=self.init_param_obj)
        # last_weight = w
        if self.role == consts.GUEST:
            self.labels = data_instances.mapValues(lambda x: np.array([x.label], dtype=int))
        source_features = data_instances.mapValues(lambda x: x.features)
        LOGGER.debug(f"source_features: {source_features.first()}")

        remote_pubkey = self.transfer_pubkey()
        LOGGER.debug(f"n: {remote_pubkey.n}")
        with SPDZ(
                "pearson",
                local_party=self.local_party,
                all_parties=self.parties,
                use_mix_rand=self.model_param.use_mix_rand,
        ) as spdz:
            self.fix_point_encoder = self.create_fixpoint_encoder(remote_pubkey.n)
            LOGGER.debug(f"fix_point_encoder: {self.fix_point_encoder.__dict__}")
            w_self, w_remote = self.share_init_model(w, self.fix_point_encoder)
            # LOGGER.debug(f"w_self: {w_self.value[0].encoding}")
            # LOGGER.debug(f"w_self: {w_self.value}, w_remote shape: {w_remote.value[0].encoding}")

            self.features = fixedpoint_table.FixedPointTensor(self.fix_point_encoder.encode(source_features),
                                                              q_field=self.fix_point_encoder.n,
                                                              endec=self.fix_point_encoder)
            LOGGER.debug(f"encoded features: {self.features.value.first()}")
            while self.n_iter_ < self.max_iter:
                LOGGER.debug(f"n_iter: {self.n_iter_}")
                current_suffix = (self.n_iter_,)
                y = self.cal_prediction(w_self, w_remote, features=self.features, spdz=spdz, suffix=current_suffix)
                if self.role == consts.GUEST:
                    error = y.value.join(self.labels, operator.sub)
                    LOGGER.debug(f"error: {error.first()}")
                    error = fixedpoint_table.FixedPointTensor.from_value(error,
                                                                         q_field=self.fix_point_encoder.n,
                                                                         encoder=self.fix_point_encoder)
                    w_remote, w_self = self.compute_gradient(wa=w_remote, wb=w_self, error=error, suffix=current_suffix)
                    LOGGER.debug(f"before_reconstruct, w_self: {w_self.value}, w_remote shape: {w_remote.value}")

                    new_w = w_self.reconstruct_unilateral(tensor_name=f"wb_{self.n_iter_}")
                    LOGGER.debug(f"new_w: {new_w}")

                    w_remote.broadcast_reconstruct_share(tensor_name=f"wa_{self.n_iter_}")

                else:
                    w_self, w_remote = self.compute_gradient(wa=w_self, wb=w_remote, error=y, suffix=current_suffix)
                    w_remote.broadcast_reconstruct_share(tensor_name=f"wb_{self.n_iter_}")
                    new_w = w_self.reconstruct_unilateral(tensor_name=f"wa_{self.n_iter_}")
                LOGGER.debug(f"new_w: {new_w}")
                self.model_weights = LinearModelWeights(l=new_w,
                                                        fit_intercept=self.model_param.init_param.fit_intercept)

                w_self, w_remote = self.share_init_model(
                    new_w, fix_point_encoder=self.fix_point_encoder, n_iter=self.n_iter_)
                # weight_diff = fate_operator.norm(last_weight - new_w)
                # if weight_diff < self.model_param.tol:
                #     self.is_converged = True
                #     break
                self.n_iter_ += 1
        self.model_weights = LinearModelWeights(l=new_w, fit_intercept=self.model_param.init_param.fit_intercept)

    def share_z(self, suffix, **kwargs):
        if self.role == consts.HOST:
            for var_name in ["z", "z_square", "z_cube"]:
                z = kwargs[var_name]
                encrypt_z = self.cipher.distribute_encrypt(z.value)
                LOGGER.debug(f"encoded_n: {encrypt_z.first()[1][0].public_key.n}")
                self.transfer_variable.encrypted_share_matrix.remote(encrypt_z, role=consts.GUEST,
                                                                     suffix=(var_name,) + suffix)
        else:
            res = []
            for var_name in ["z", "z_square", "z_cube"]:
                z_table = self.transfer_variable.encrypted_share_matrix.get_parties(
                    self.other_party,
                    suffix=(var_name,) + suffix)[0]
                res.append(fixedpoint_table.PaillierFixedPointTensor(
                    z_table, q_field=self.features.q_field, endec=self.features.endec))
            return res[0], res[1], res[2]

    def _get_param(self):
        param_protobuf_obj = lr_model_param_pb2.LRModelParam()
        return param_protobuf_obj

    def _get_meta(self):
        meta_protobuf_obj = lr_model_meta_pb2.LRModelMeta(penalty=self.model_param.penalty,
                                                          tol=self.model_param.tol,
                                                          alpha=self.alpha,
                                                          optimizer=self.model_param.optimizer,
                                                          batch_size=self.batch_size,
                                                          learning_rate=self.model_param.learning_rate,
                                                          max_iter=self.max_iter,
                                                          early_stop=self.model_param.early_stop,
                                                          fit_intercept=self.fit_intercept,
                                                          need_one_vs_rest=self.need_one_vs_rest)
        return meta_protobuf_obj
