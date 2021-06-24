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
#

import functools
import math
import sys

import numpy as np

from fate_arch.session import is_table
from federatedml.util import LOGGER


class FixedPointNumber(object):
    """Represents a float or int fixedpoit encoding;.
    """
    BASE = 16
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    Q = 293973345475167247070445277780365744413

    def __init__(self, encoding, exponent, n=None, max_int=None):
        self.n = n
        self.max_int = max_int

        if self.n is None:
            self.n = self.Q
            self.max_int = self.Q // 3 - 1

        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, scalar, n=None, max_int=None, precision=None, max_exponent=None):
        """return an encoding of an int or float.
        """
        # Calculate the maximum exponent for desired precision
        exponent = None

        #  Too low value preprocess;
        #  avoid "OverflowError: int too large to convert to float"
        # LOGGER.debug(f"In encode, scalar: {scalar}")

        if np.abs(scalar) < 1e-200:
            scalar = 0

        # if isinstance(scalar, int) and scalar.bit_length() > 70:
        #     raise OverflowError(f"scalar: {scalar}, n: {n}")

        if n is None:
            n = cls.Q
            max_int = cls.Q // 3 - 1

        if precision is None:
            if isinstance(scalar, int) or isinstance(scalar, np.int16) or \
                    isinstance(scalar, np.int32) or isinstance(scalar, np.int64):
                exponent = 0
            elif isinstance(scalar, float) or isinstance(scalar, np.float16) \
                    or isinstance(scalar, np.float32) or isinstance(scalar, np.float64):
                flt_exponent = math.frexp(scalar)[1]
                lsb_exponent = cls.FLOAT_MANTISSA_BITS - flt_exponent
                exponent = math.floor(lsb_exponent / cls.LOG2_BASE)
            else:
                raise TypeError("Don't know the precision of type %s."
                                % type(scalar))
        else:
            exponent = math.floor(math.log(precision, cls.BASE))

        if max_exponent is not None:
            exponent = max(max_exponent, exponent)

        int_fixpoint = int(round(scalar * pow(cls.BASE, exponent)))

        if abs(int_fixpoint) > max_int:
            raise ValueError('Integer needs to be within +/- %d but got %d'
                             % (max_int, int_fixpoint))
        # if exponent < 0:
        #     raise ValueError(f"exponent: {exponent}, scalar: {scalar}, n: {n}")
        # LOGGER.debug(f"In encode, scalar: {scalar}, exponent: {exponent}, encoding: {int_fixpoint % n}")
        # if int_fixpoint % n == 0:
        #     assert 1 == 2

        return cls(int_fixpoint % n, exponent, n, max_int)

    def decode(self):
        """return decode plaintext.
        """
        if self.encoding >= self.n:
            # Should be mod n
            raise ValueError('Attempted to decode corrupted number')
        elif self.encoding <= self.max_int:
            # Positive
            mantissa = self.encoding
        elif self.encoding >= self.n - self.max_int:
            # Negative
            mantissa = self.encoding - self.n
        else:
            LOGGER.debug(f"In decode, exponent: {self.exponent}, bit_lenght: {self.exponent.bit_length()}，"
                         f"encoding: {self.encoding.bit_length()}")
            raise OverflowError('Overflow detected in decode number')
        # LOGGER.debug(f"In decode, exponent: {self.exponent}, bit_lenght: {self.exponent.bit_length()},"
        #              f"mantissa: {mantissa}, base: {self.BASE}")
        return mantissa * pow(self.BASE, -self.exponent)

    def increase_exponent_to(self, new_exponent):
        """return FixedPointNumber: new encoding with same value but having great exponent.
        """
        if new_exponent < self.exponent:
            raise ValueError('New exponent %i should be greater than'
                             'old exponent %i' % (new_exponent, self.exponent))

        factor = pow(self.BASE, new_exponent - self.exponent)
        new_encoding = self.encoding * factor % self.n

        return FixedPointNumber(new_encoding, new_exponent, self.n, self.max_int)

    def __align_exponent(self, x, y):
        """return x,y with same exponet
        """
        if x.exponent < y.exponent:
            x = x.increase_exponent_to(y.exponent)
        elif x.exponent > y.exponent:
            y = y.increase_exponent_to(x.exponent)

        return x, y

    def __truncate(self, a):
        scalar = a.decode()
        return FixedPointNumber.encode(scalar, n=self.n, max_int=self.max_int)

    def __add__(self, other):
        if isinstance(other, FixedPointNumber):
            return self.__add_fixpointnumber(other)
        else:
            return self.__add_scalar(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, FixedPointNumber):
            return self.__sub_fixpointnumber(other)
        else:
            return self.__sub_scalar(other)

    def __rsub__(self, other):
        x = self.__sub__(other)
        x = -1 * x.decode()
        return self.encode(x, n=self.n, max_int=self.max_int)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        try:
            if isinstance(other, FixedPointNumber):
                return self.__mul_fixpointnumber(other)
            else:
                return self.__mul_scalar(other)
        except:
            encoding = (self.encoding * other.encoding) % self.n
            exponet = self.exponent + other.exponent
            mul_fixedpoint = FixedPointNumber(encoding, exponet, n=self.n, max_int=self.max_int)
            # truncate_mul_fixedpoint = self.__truncate(mul_fixedpoint)
            raise OverflowError(f"other: {other.encoding}, self.encoding: {self.encoding},"
                                f"encoding: {encoding}, exponent: {exponet}"
                                f"mul_fixedpoint: {mul_fixedpoint.encoding},"
                                f"mul_fixedpoint exponent: {mul_fixedpoint.exponent},"
                                f"n: {self.n}")

    def __truediv__(self, other):
        if isinstance(other, FixedPointNumber):
            scalar = other.decode()
        else:
            scalar = other

        return self.__mul__(1 / scalar)

    def __rtruediv__(self, other):
        res = 1.0 / self.__truediv__(other).decode()
        return FixedPointNumber.encode(res, n=self.n, max_int=self.max_int)

    def __lt__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):
            y = other.decode()
        else:
            y = other
        if x < y:
            return True
        else:
            return False

    def __gt__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):
            y = other.decode()
        else:
            y = other
        if x > y:
            return True
        else:
            return False

    def __le__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):
            y = other.decode()
        else:
            y = other
        if x <= y:
            return True
        else:
            return False

    def __ge__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):
            y = other.decode()
        else:
            y = other

        if x >= y:
            return True
        else:
            return False

    def __eq__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):
            y = other.decode()
        else:
            y = other
        if x == y:
            return True
        else:
            return False

    def __ne__(self, other):
        x = self.decode()
        if isinstance(other, FixedPointNumber):
            y = other.decode()
        else:
            y = other
        if x != y:
            return True
        else:
            return False

    def __add_fixpointnumber(self, other):
        if self.n != other.n:
            # raise ValueError(f"Adding number with different field")
            other = self.encode(other.decode(), n=self.n, max_int=self.max_int)
        x, y = self.__align_exponent(self, other)
        encoding = (x.encoding + y.encoding) % self.n
        from federatedml.util import LOGGER
        LOGGER.debug(f"in __add_fixpointnumber, encoding: {encoding}, n: {self.n}")
        added_num = FixedPointNumber(encoding, x.exponent, n=self.n, max_int=self.max_int)
        # if added_num.exponent < 0:
        #     raise ValueError(f"exponent: {added_num.exponent}")
        # if added_num.encoding.bit_length() > 150 and \
        #         (self.n - added_num.encoding).bit_length() > 150:
        #     raise OverflowError(f"other: {other.encoding}, self.encoding: {self.encoding},"
        #                         f"exponent: {other.exponent}, {self.exponent}"
        #                         f"x.encoding: {x.encoding}, exponent: {x.exponent}"
        #                         f"y.encoding: {y.encoding}, exponent: {y.exponent}"
        #                         f"encoding: {encoding}, exponent: {added_num.exponent}"
        #                         f"mul_fixedpoint: {added_num.encoding},"
        #                         f"mul_fixedpoint exponent: {added_num.exponent},"
        #                         f"n: {self.n},"
        #                         f"added_num.encoding.bit_length(): {added_num.encoding.bit_length()},"
        #                         f"(self.n - added_num.encoding).bit_length():"
        #                         f"{(self.n - added_num.encoding).bit_length()}")
        return self.__truncate(added_num)

    def __add_scalar(self, scalar):
        encoded = self.encode(scalar, n=self.n, max_int=self.max_int)
        return self.__add_fixpointnumber(encoded)

    def __sub_fixpointnumber(self, other):
        scalar = -1 * other.decode()
        return self.__add_scalar(scalar)

    def __sub_scalar(self, scalar):
        scalar = -1 * scalar
        return self.__add_scalar(scalar)

    def __mul_fixpointnumber(self, other):
        if self.n != other.n:
            raise ValueError(f"Multiplying number with different field")
        encoding = (self.encoding * other.encoding) % self.n
        exponet = self.exponent + other.exponent
        mul_fixedpoint = FixedPointNumber(encoding, exponet, n=self.n, max_int=self.max_int)
        truncate_mul_fixedpoint = self.__truncate(mul_fixedpoint)
        # if truncate_mul_fixedpoint.encoding.bit_length() > 150 and \
        #         (self.n - truncate_mul_fixedpoint.encoding).bit_length() > 150:
        #     raise ValueError()
        # if exponet < 0:
        #     raise ValueError(f"exponent: {exponet}")
        return truncate_mul_fixedpoint

    def __mul_scalar(self, scalar):
        encoded = self.encode(scalar, n=self.n, max_int=self.max_int)
        return self.__mul_fixpointnumber(encoded)

    def __abs__(self):
        if self.encoding <= self.max_int:
            # Positive
            return self
        elif self.encoding >= self.n - self.max_int:
            # Negative
            return self * -1


class FixedPointEndec(object):
    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    @staticmethod
    def table_op(x, op):
        arr = np.zeros(shape=x.shape, dtype=object)
        view = arr.view().reshape(-1)
        x_array = x.view().reshape(-1)
        for i in range(arr.size):
            view[i] = op(x_array[i])
        return arr

    @staticmethod
    def table_decode_op(x):
        arr = np.zeros(shape=x.shape, dtype=object)
        view = arr.view().reshape(-1)
        for i in range(arr.size):
            view[i] = view[i].decode()
        return arr

    @classmethod
    def _basic_op(cls, tensor, op):
        if isinstance(tensor, np.ndarray):
            arr = np.zeros(shape=tensor.shape, dtype=object)
            view = arr.view().reshape(-1)
            t = tensor.view().reshape(-1)
            for i in range(arr.size):
                LOGGER.debug(f"t: {t}, ti: {t[i]}, op: {op}")
                view[i] = op(t[i])
            return arr

        elif is_table(tensor):
            LOGGER.debug(f"is_table, tensor: {tensor.first()}")
            f = functools.partial(cls.table_op, op=op)
            return tensor.mapValues(f)
        else:
            return op(tensor)

    def encode(self, float_tensor):
        f = functools.partial(FixedPointNumber.encode,
                              n=self.n, max_int=self.max_int)
        return self._basic_op(float_tensor, op=f)

    def __truncate_op(self, a):
        scalar = a.decode()
        return FixedPointNumber.encode(scalar, n=self.n, max_int=self.max_int)

    @staticmethod
    def decode_number(number):
        return number.decode()

    def decode(self, integer_tensor):
        return self._basic_op(integer_tensor, op=self.decode_number)

    def truncate(self, integer_tensor):
        return self._basic_op(integer_tensor, op=self.__truncate_op)
