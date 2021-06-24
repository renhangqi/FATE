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

import numpy as np
from fate_arch.session import is_table


class FixedPointEndec(object):

    def __init__(self, field: int, base: int, precision_fractional: int):
        self.field = field
        self.base = base
        self.precision_fractional = precision_fractional

    def decode(self, integer_tensor: np.ndarray):
        value = integer_tensor % self.field
        gate = value > self.field // 2
        neg_nums = (value - self.field) * gate
        pos_nums = value * (1 - gate)
        result = (neg_nums + pos_nums) / (self.base ** self.precision_fractional)
        return result

    def encode(self, float_tensor, check_range=True):
        if isinstance(float_tensor, (float, np.float)):
            float_tensor = np.array(float_tensor)
        if isinstance(float_tensor, np.ndarray):
            upscaled = (float_tensor * self.base ** self.precision_fractional).astype(np.int64)
            if check_range:
                assert (np.abs(upscaled) < (self.field / 2)).all(), (
                    f"{float_tensor} cannot be correctly embedded: choose bigger field or a lower precision"
                )

            field_element = upscaled % self.field
            return field_element
        elif is_table(float_tensor):
            s = self.base ** self.precision_fractional
            upscaled = float_tensor.mapValues(lambda x: (x * s).astype(np.int64))
            if check_range:
                assert upscaled.filter(lambda k, v: (np.abs(v) >= self.field / 2).any()).count() == 0, (
                    f"{float_tensor} cannot be correctly embedded: choose bigger field or a lower precision"
                )
            field_element = upscaled.mapValues(lambda x: x % self.field)
            return field_element
        else:
            raise ValueError(f"unsupported type: {type(float_tensor)}")

    def truncate(self, integer_tensor, idx=0):
        if idx == 0:
            return self.field - (self.field - integer_tensor) // (self.base ** self.precision_fractional)
        else:
            return integer_tensor // (self.base ** self.precision_fractional)


class FixedPointObjectEndec(FixedPointEndec):

    # def encode(self, float_tensor, check_range=True):
    #     return float_tensor
    #
    # def decode(self, integer_tensor: np.ndarray):
    #     return integer_tensor

    def encode(self, float_tensor, check_range=True):
        if isinstance(float_tensor, (float, np.float)):
            float_tensor = np.array(float_tensor)
        if isinstance(float_tensor, np.ndarray):
            upscaled = (float_tensor * self.base ** self.precision_fractional).astype(np.int64).astype(object)
            if check_range:
                assert (np.abs(upscaled) < (self.field // 2)).all(), (
                    f"{float_tensor} cannot be correctly embedded: choose bigger field or a lower precision"
                )

            field_element = upscaled % self.field
            return field_element
        elif is_table(float_tensor):
            s = self.base ** self.precision_fractional
            upscaled = float_tensor.mapValues(lambda x: (x * s).astype(np.int64).astype(object))
            if check_range:
                assert upscaled.filter(lambda k, v: (np.abs(v) >= self.field / 2).any()).count() == 0, (
                    f"{float_tensor} cannot be correctly embedded: choose bigger field or a lower precision"
                )
            field_element = upscaled.mapValues(lambda x: x % self.field)
            return field_element
        else:
            raise ValueError(f"unsupported type: {type(float_tensor)}")


endec = FixedPointObjectEndec(2 << 1024, 10, 4)
a = np.array([-1.447])
aa = endec.encode(a)
b = endec.decode(aa*aa)
