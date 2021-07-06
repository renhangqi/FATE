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
from federatedml.optim import activation
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.util import LOGGER, consts
from federatedml.util import fate_operator
from federatedml.util.io_check import assert_io_num_rows_equal
import numpy as np


class HeteroLRGuest(HeteroLRBase):

    def __init__(self):
        super().__init__()
        self.data_batch_count = []

    def transfer_pubkey(self):
        public_key = self.cipher.public_key
        self.transfer_variable.pubkey.remote(public_key, role=consts.HOST, suffix=("guest_pubkey",))
        remote_pubkey = self.transfer_variable.pubkey.get_parties(parties=self.other_party,
                                                                  suffix=("host_pubkey",))[0]
        return remote_pubkey

    def cal_prediction(self, w_self, w_remote, features, spdz, suffix):
        # n = features.value.count()
        z1 = features.dot_array(w_self.value, fit_intercept=self.fit_intercept)

        za_suffix = ("za",) + suffix
        self.secure_matrix_mul_active(w_remote, cipher=self.cipher, suffix=za_suffix)
        # z1 = features.dot_array(w_self.value)
        za_share = self.received_share_matrix(self.cipher, q_field=self.fix_point_encoder.n,
                                              encoder=self.fix_point_encoder, suffix=za_suffix)
        zb_share = self.secure_matrix_mul_passive(features,
                                                  suffix=("zb",) + suffix)
        z = z1 + za_share + zb_share

        # z = z.convert_to_array_tensor()
        # new_w = z.reconstruct_unilateral(tensor_name=f"z_{self.n_iter_}")
        # raise ValueError(f"reconstructed z: {new_w}")

        # z_square = z * z
        # z_cube = z_square * z

        remote_z, remote_z_square, remote_z_cube = self.share_z(suffix=suffix)

        complete_z = remote_z + z
        # complete_z_cube = remote_z_cube + remote_z_square * z * 3 + remote_z * z_square * 3 + z_cube
        # sigmoid_z = complete_z * 0.197 - complete_z_cube * 0.004 + 0.5
        sigmoid_z = complete_z * 0.2 + 0.5

        shared_sigmoid_z = self.share_matrix(sigmoid_z, suffix=("sigmoid_z",) + suffix)
        return shared_sigmoid_z

    def compute_gradient(self, wa, wb, error, features, suffix):
        LOGGER.debug(f"start_wa shape: {wa.value}, wb shape: {wb.value}")

        n = error.value.count()
        encoded_1_n = self.fix_point_encoder.encode(1 / n)
        error_1_n = error * encoded_1_n
        encrypt_error = fixedpoint_table.PaillierFixedPointTensor.from_value(
            self.transfer_variable.share_error.get(idx=0, suffix=suffix),
            q_field=self.fix_point_encoder.n,
            encoder=self.fix_point_encoder
        )
        encrypt_error = encrypt_error + error

        encrypt_g = encrypt_error.value.join(features.value, operator.mul).reduce(operator.add) * encoded_1_n
        if self.fit_intercept:
            bias = encrypt_error.value.reduce(operator.add) * encoded_1_n
            encrypt_g = np.array(list(encrypt_g) + list(bias))
        encrypt_g = fixedpoint_numpy.PaillierFixedPointTensor(encrypt_g, q_field=error.q_field,
                                                              endec=self.fix_point_encoder)
        gb2 = self.share_matrix(encrypt_g, suffix=("encrypt_g",) + suffix)

        ga2_suffix = ("ga2",) + suffix
        self.secure_matrix_mul_active(error_1_n, cipher=self.cipher,
                                      suffix=ga2_suffix)
        ga2_2 = self.received_share_matrix(self.cipher, q_field=self.fix_point_encoder.n,
                                           encoder=self.fix_point_encoder, suffix=ga2_suffix)
        # if self.fit_intercept:
        #     bias = error.value.reduce(operator.add) * encoded_1_n
        #     ga2_2.value = np.array(list(ga2_2.value) + list(bias))
        wb = wb - gb2 * self.model_param.learning_rate
        wa = wa - ga2_2.transpose() * self.model_param.learning_rate
        wa = wa.reshape(wa.shape[-1])

        return wa, wb

    def check_converge(self, last_w, new_w, suffix):
        square_sum = np.sum((last_w - new_w) ** 2)
        host_sums = self.converge_transfer_variable.square_sum.get(suffix=suffix)
        for hs in host_sums:
            square_sum += hs
        norm_diff = np.sqrt(square_sum)
        is_converge = False
        if norm_diff < self.model_param.tol:
            is_converge = True
        self.converge_transfer_variable.converge_info.remote(is_converge, role=consts.HOST, suffix=suffix)
        return is_converge

    @assert_io_num_rows_equal
    def predict(self, data_instances):
        """
        Prediction of lr
        Parameters
        ----------
        data_instances: DTable of Instance, input data

        Returns
        ----------
        DTable
            include input data label, predict probably, label
        """
        LOGGER.info("Start predict is a one_vs_rest task: {}".format(self.need_one_vs_rest))
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)

        pred_prob = data_instances.mapValues(lambda v: fate_operator.vec_dot(v.features, self.model_weights.coef_)
                                                       + self.model_weights.intercept_)
        # pred_prob = self.compute_wx(data_instances, self.model_weights.coef_, self.model_weights.intercept_)
        host_probs = self.transfer_variable.host_prob.get(idx=-1)

        LOGGER.info("Get probability from Host")

        # guest probability
        for host_prob in host_probs:
            pred_prob = pred_prob.join(host_prob, lambda g, h: g + h)
        pred_prob = pred_prob.mapValues(lambda p: activation.sigmoid(p))
        threshold = self.model_param.predict_param.threshold
        predict_result = self.predict_score_to_output(data_instances, pred_prob, classes=[0, 1], threshold=threshold)

        return predict_result
