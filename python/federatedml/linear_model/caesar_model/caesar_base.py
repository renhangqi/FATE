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
from abc import ABC

import numpy as np

from fate_arch import session
from federatedml.linear_model.linear_model_base import BaseLinearModel
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.transfer_variable.transfer_class.caesar_model_transfer_variable import CaesarModelTransferVariable
from federatedml.util import consts, LOGGER


class CaesarBase(BaseLinearModel, ABC):
    def __init__(self):
        super().__init__()
        self._set_parties()
        self.transfer_variable = CaesarModelTransferVariable()

    def _set_parties(self):
        # since multi-host not supported yet, we assume parties are one from guest and one from host
        parties = []
        guest_parties = session.get_latest_opened().parties.roles_to_parties(["guest"])
        host_parties = session.get_latest_opened().parties.roles_to_parties(["host"])
        if len(guest_parties) != 1 or len(host_parties) != 1:
            raise ValueError(
                f"one guest and one host required, "
                f"while {len(guest_parties)} guest and {len(host_parties)} host provided"
            )
        parties.extend(guest_parties)
        parties.extend(host_parties)

        local_party = session.get_latest_opened().parties.local_party
        other_party = parties[0] if parties[0] != local_party else parties[1]

        self.parties = parties
        self.local_party = local_party
        self.other_party = other_party

    @staticmethod
    def create_fixpoint_encoder(spdz, **kwargs):
        base = kwargs['base'] if 'base' in kwargs else 10
        frac = kwargs['frac'] if 'frac' in kwargs else 4
        q_field = kwargs['q_field'] if 'q_field' in kwargs else spdz.q_field
        encoder = fixedpoint_numpy.FixedPointObjectEndec(q_field, base, frac)
        return encoder

    def share_matrix(self, matrix_tensor, suffix=tuple()):
        curt_suffix = ("share_matrix",) + suffix
        table = matrix_tensor.value
        r = fixedpoint_table.urand_tensor(q_field=matrix_tensor.q_field,
                                          tensor=table)
        if isinstance(matrix_tensor, fixedpoint_table.FixedPointTensor):
            random_tensor = fixedpoint_table.FixedPointTensor.from_value(value=r,
                                                                         q_field=matrix_tensor.q_field,
                                                                         endec=matrix_tensor.endec)
            to_share = matrix_tensor.value.join(random_tensor.value, operator.sub)
        elif isinstance(matrix_tensor, fixedpoint_numpy.FixedPointTensor):
            random_tensor = fixedpoint_numpy.FixedPointTensor(value=r,
                                                              q_field=matrix_tensor.q_field,
                                                              endec=matrix_tensor.endec)
            to_share = (matrix_tensor - random_tensor).value
        else:
            raise ValueError(f"Share_matrix input error, type of input: {type(matrix_tensor)}")
        # to_share = matrix_tensor - random_tensor
        dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST
        self.transfer_variable.share_matrix.remote(to_share, role=dest_role, suffix=curt_suffix)
        return random_tensor

    def received_share_matrix(self, cipher, q_field, encoder, suffix=tuple()):
        curt_suffix = ("share_matrix",) + suffix
        share = self.transfer_variable.share_matrix.get_parties(parties=self.other_party,
                                                                suffix=curt_suffix)[0]
        if isinstance(share, np.ndarray):
            share = cipher.recursive_decrypt(share)
            LOGGER.debug(f"received_share: {share}")
            return fixedpoint_numpy.FixedPointTensor(value=share,
                                                     q_field=q_field,
                                                     endec=encoder)
        share = cipher.distribute_decrypt(share)
        # share = share.mapValues(lambda x: x % q_field)
        return fixedpoint_table.FixedPointTensor.from_value(share, q_field=q_field, encoder=encoder)

    def secure_matrix_mul(self, matrix, cipher=None, suffix=tuple()):
        curt_suffix = ("secure_matrix_mul",) + suffix
        if cipher is not None:  # Host
            dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST
            # if isinstance(matrix, fixedpoint_table.FixedPointTensor):
            #     encrypt_mat = cipher.distribute_encrypt(matrix.value)
            # else:
            #     encrypt_mat = cipher.recursive_decrypt(matrix.value)
            encrypt_mat = cipher.distribute_encrypt(matrix.value)
            self.transfer_variable.share_matrix.remote(encrypt_mat, role=dest_role, idx=0, suffix=curt_suffix)
            return self.received_share_matrix(cipher, q_field=matrix.q_field, encoder=matrix.endec, suffix=suffix)
        else:  # Guest
            share = self.transfer_variable.share_matrix.get_parties(parties=self.other_party,
                                                                    suffix=curt_suffix)[0]
            share_tensor = fixedpoint_table.PaillierFixedPointTensor.from_value(
                share, q_field=matrix.q_field, encoder=matrix.endec)

            LOGGER.debug(f"Make share tensor")
            if isinstance(matrix, fixedpoint_numpy.FixedPointTensor):
                xy = share_tensor.dot_array(matrix.value)
            else:
                xy = share_tensor.dot_local(matrix)
            LOGGER.debug(f"Finish dot")
            # xy_tensor = matrix.from_value(xy, q_field=matrix.q_field, encoder=matrix.endec)
            return self.share_matrix(xy, suffix=suffix)
