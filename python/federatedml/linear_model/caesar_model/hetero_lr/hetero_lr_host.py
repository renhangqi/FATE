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
import numpy as np


class HeteroLRHost(HeteroLRBase):
    def __init__(self):
        super().__init__()
        self.data_batch_count = []

    def transfer_pubkey(self):
        public_key = self.cipher.public_key
        self.transfer_variable.pubkey.remote(public_key, role=consts.GUEST, suffix=("host_pubkey",))
        remote_pubkey = self.transfer_variable.pubkey.get_parties(parties=self.other_party,
                                                                  suffix=("guest_pubkey",))[0]
        return remote_pubkey

    def cal_prediction(self, w_self, w_remote, features, spdz, suffix):
        # za_share = self.secure_matrix_mul(self.features, cipher=self.cipher, suffix=("za",) + suffix)
        # zb_share = self.secure_matrix_mul(w_remote, suffix=("zb",) + suffix)
        z1 = features.dot_array(w_self.value)
        za_share = self.secure_matrix_mul_passive(features,  suffix=("za",) + suffix)

        zb_suffix = ("zb",) + suffix
        self.secure_matrix_mul_active(w_remote, cipher=self.cipher, suffix=zb_suffix)
        zb_share = self.received_share_matrix(self.cipher, q_field=self.fix_point_encoder.n,
                                              encoder=self.fix_point_encoder, suffix=zb_suffix)
        z = z1 + za_share + zb_share

        self.share_z(suffix=suffix, z=z, z_square=z, z_cube=z)

        shared_sigmoid_z = self.received_share_matrix(self.cipher,
                                                      q_field=z.q_field,
                                                      encoder=z.endec,
                                                      suffix=("sigmoid_z",) + suffix)
        return shared_sigmoid_z

    def compute_gradient(self, wa, wb, error: fixedpoint_table.FixedPointTensor, features, suffix):
        n = error.value.count()
        encrypt_error = self.cipher.distribute_encrypt(error.value)
        self.transfer_variable.share_error.remote(encrypt_error, idx=0, role=consts.GUEST, suffix=suffix)
        gb1 = self.received_share_matrix(cipher=self.cipher, q_field=error.q_field,
                                         encoder=error.endec, suffix=("encrypt_g",) + suffix)
        LOGGER.debug(f"gb1_value: {gb1.value}")
        encoded_1_n = self.fix_point_encoder.encode(1 / n)
        ga = error.value.join(features.value, operator.mul).reduce(operator.add) * encoded_1_n
        ga = fixedpoint_numpy.FixedPointTensor(ga, q_field=error.q_field,
                                               endec=self.fix_point_encoder)
        ga2_1 = self.secure_matrix_mul_passive(features, suffix=("ga2",) + suffix)
        LOGGER.debug(f"ga: {ga.value}, ga2_1: {ga2_1.value}")
        learning_rate = self.fix_point_encoder.encode(self.model_param.learning_rate)
        # learning_rate = self.model_param.learning_rate
        ga_new = ga + ga2_1.reshape(ga2_1.shape[0])

        wa = wa - ga_new.transpose() * learning_rate
        wb = wb - gb1 * learning_rate
        wa = wa.reshape(wa.shape[-1])

        return wa, wb

    def check_converge(self, last_w, new_w, suffix):
        square_sum = np.sum((last_w - new_w) ** 2)
        self.converge_transfer_variable.square_sum.remote(square_sum, role=consts.GUEST, idx=0, suffix=suffix)
        return self.converge_transfer_variable.converge_info.get(idx=0, suffix=suffix)

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
