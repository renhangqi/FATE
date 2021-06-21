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

from federatedml.linear_model.caesar_model.hetero_lr.hetero_lr_base import HeteroLRBase
from federatedml.secureprotol.spdz.tensor import fixedpoint_table, fixedpoint_numpy
from federatedml.util import consts, LOGGER
from federatedml.util import fate_operator


class HeteroLRHost(HeteroLRBase):
    def __init__(self):
        super().__init__()
        self.data_batch_count = []

    def cal_prediction(self, w_self, w_remote, features, spdz, suffix):
        z1 = features.dot_array(w_self.value)
        za_share = self.secure_matrix_mul(features, cipher=self.cipher, suffix=("za",) + suffix)
        zb_share = self.secure_matrix_mul(w_remote, suffix=("zb",) + suffix)
        LOGGER.debug(f"w_x: {z1.value.first()}, za_share: {za_share.value.first()},"
                     f" zb_share: {zb_share.value.first()}")

        z = z1 + za_share + zb_share
        z_square = z * z
        z_cube = z_square * z
        LOGGER.debug(f"cal_prediction z: {z.value.first()}, z_square: {z_square.value.first()},"
                     f"z_cube: {z_cube.value.first()}")
        self.share_z(suffix=(self.n_iter_,), z=z, z_square=z_square, z_cube=z_cube)
        shared_sigmoid_z = self.received_share_matrix(self.cipher,
                                                      q_field=z.q_field,
                                                      encoder=z.endec,
                                                      suffix=("sigmoid_z",) + suffix)
        LOGGER.debug(f"shared_sigmoid_z: {list(shared_sigmoid_z.value.collect())}")
        return shared_sigmoid_z

    def compute_gradient(self, wa, wb, error: fixedpoint_table.FixedPointTensor, suffix):
        n = error.value.count()
        LOGGER.debug(f"In compute_gradient, error: {list(error.value.collect())}")
        encrypt_error = self.cipher.distribute_encrypt(error.value)
        self.transfer_variable.share_error.remote(encrypt_error, idx=0, role=consts.GUEST, suffix=suffix)
        gb1 = self.received_share_matrix(cipher=self.cipher, q_field=error.q_field,
                                         encoder=error.endec, suffix=("encrypt_g",) + suffix)
        LOGGER.debug(f"gb1_value: {gb1.value}")
        encoded_1_n = self.fix_point_encoder.encode(1 / n)
        ga = error.value.join(self.features.value, operator.mul).reduce(operator.add) * encoded_1_n
        ga = fixedpoint_numpy.FixedPointTensor(ga, q_field=error.q_field,
                                               endec=self.fix_point_encoder)
        # ga = error.dot_local(self.features) * encoded_1_n
        ga2_1 = self.secure_matrix_mul(self.features, cipher=self.cipher, suffix=("ga2",) + suffix)
        LOGGER.debug(f"ga: {ga.value}, ga2_1: {ga2_1.value}")
        learning_rate = self.fix_point_encoder.encode(self.model_param.learning_rate)
        ga_new = ga + ga2_1.reshape(ga2_1.shape[0])
        LOGGER.debug(f"ga_shape: {ga.shape}, ga2_1.shape: {ga2_1.shape}, ga_shape: {ga_new.shape}")

        wa = wa - learning_rate * ga_new.transpose()
        wb = wb - learning_rate * gb1
        wa = wa.reshape(wa.shape[-1])
        LOGGER.debug(f"wa shape: {wa.value}, wb shape: {wb.value}, gb1.shape: {gb1.value},"
                     f"ga_new.shape: {ga_new.value}")
        LOGGER.debug(f"wa shape: {wa.value.shape}, wb shape: {wb.value.shape}, ga_shape: {ga.shape}")

        return wa, wb

    def predict(self, data_instances):
        self.transfer_variable.host_prob.disable_auto_clean()
        LOGGER.info("Start predict ...")
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        prob_host = data_instances.mapValues(lambda v: fate_operator.vec_dot(v.features, self.model_weights.coef_)
                                                        + self.model_weights.intercept_)
        # prob_host = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)
        self.transfer_variable.host_prob.remote(prob_host, role=consts.GUEST, idx=0)
        LOGGER.info("Remote probability to Guest")
