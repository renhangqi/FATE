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
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.util import LOGGER, consts
from federatedml.util import fate_operator
from federatedml.optim import activation
from federatedml.util.io_check import assert_io_num_rows_equal


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
        z1 = features.dot_array(w_self.value)
        LOGGER.debug(f"features: {features.value.first()}, n: {features.value.first()[1][0].n},"
                     f"w_self: {w_self.value[0].n}, z1: {z1.value.first()[1][0].n}")
        # LOGGER.debug(f"before mul, w_remote: {w_remote.value[0].encoding}, {w_remote.value[0].exponent}")
        za_share = self.secure_matrix_mul(w_remote, suffix=("za",) + suffix)
        zb_share = self.secure_matrix_mul(features, cipher=self.cipher, suffix=("zb",) + suffix)

        z = z1 + za_share + zb_share
        LOGGER.debug(f"z1.value.first()")
        LOGGER.debug(f"zzzz: {z1.value.first()[1][0].n},"
                     f" {za_share.value.first()},"
                     f" {zb_share.value.first()},"
                     f" {z.value.first()[1][0].n}")
        z_square = z * z
        z_cube = z_square * z
        LOGGER.debug(f"cal_prediction z: {z.value.first()}, z_square: {z_square.value.first()},"
                     f"z_cube: {z_cube.value.first()}")
        remote_z, remote_z_square, remote_z_cube = self.share_z(suffix=suffix)
        LOGGER.debug(f"remote_z_type: {type(remote_z)}")
        LOGGER.debug(f"z_ns: {z.value.first()}, n: {z.value.first()[1][0].n},"
                     f" remote_z: {remote_z.value.first()[1][0].public_key.n}")

        complete_z = remote_z + z
        # complete_z_cube = remote_z_cube + remote_z_square * z * 3 + remote_z * z_square * 3 + z_cube
        # LOGGER.debug(f"complete_z_cube count: {complete_z_cube.value.count()}")
        # sigmoid_z = complete_z * 0.197 - complete_z_cube * 0.004 + 0.5
        sigmoid_z = complete_z * 0.2 + 0.5

        shared_sigmoid_z = self.share_matrix(sigmoid_z, suffix=("sigmoid_z",) + suffix)
        LOGGER.debug(f"shared_sigmoid_z: {list(shared_sigmoid_z.value.collect())}")
        return shared_sigmoid_z

    def compute_gradient(self, wa, wb, error, suffix):
        LOGGER.debug(f"start_wa shape: {wa.value}, wb shape: {wb.value}")

        n = error.value.count()
        encrypt_error = fixedpoint_table.PaillierFixedPointTensor.from_value(
            self.transfer_variable.share_error.get(idx=0, suffix=suffix),
            q_field=self.fix_point_encoder.n,
            encoder=self.fix_point_encoder
        )
        encrypt_error = encrypt_error + error
        encoded_1_n = self.fix_point_encoder.encode(1 / n)

        encrypt_g = encrypt_error.value.join(self.features.value, operator.mul).reduce(operator.add) * encoded_1_n
        encrypt_g = fixedpoint_numpy.PaillierFixedPointTensor(encrypt_g, q_field=error.q_field,
                                                              endec=self.fix_point_encoder)
        # encrypt_g = encrypt_error.dot_local(self.features) * encoded_1_n
        gb2 = self.share_matrix(encrypt_g, suffix=("encrypt_g",) + suffix)
        # gb2.value = self.fix_point_encoder.encode(gb2.value)

        ga2_2 = self.secure_matrix_mul(error * encoded_1_n, suffix=("ga2",) + suffix)
        # ga2_2.value = self.fix_point_encoder.encode(ga2_2.value)

        # learning_rate = self.fix_point_encoder.encode(self.model_param.learning_rate)
        learning_rate = self.model_param.learning_rate

        LOGGER.debug(f"before sub, start_wa shape: {wa.value}, wb shape: {wb.value}, learning_rate: {learning_rate},"
                     f"gb2: {gb2.value}, ga2_2: {ga2_2}")
        # wb.value = self.fix_point_encoder.truncate(wb.value)
        gb2 = gb2 * learning_rate
        # gb2.value = self.fix_point_encoder.truncate(gb2.value)

        wb = wb - gb2
        # wb.value = self.fix_point_encoder.truncate(wb.value)

        wa = wa - ga2_2.transpose() * learning_rate
        wa = wa.reshape(wa.shape[-1])
        LOGGER.debug(f"wa shape: {wa.value}, wb shape: {wb.value}, gb2.shape: {gb2.value},"
                     f"ga2_2.shape: {ga2_2.value}")
        return wa, wb

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
