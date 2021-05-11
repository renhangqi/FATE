from federatedml.secureprotol.fixedpoint import FixedPointNumber
from federatedml.secureprotol import PaillierEncrypt, IterativeAffineEncrypt
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitter import SplitInfo
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.g_h_optim import *
import numpy as np
import time

sample_num = 1000
g = np.concatenate([-np.random.random(sample_num)])
h = np.random.random(sample_num)

encrypter = IterativeAffineEncrypt()
encrypter.generate_key(1024)
precision = 2 ** 53
pos_max, neg_min = get_homo_encryption_max_int(encrypter)
g_assign_bit, g_modulo, g_max_int, h_assign_bit, h_modulo, h_max_int, capacity = bit_assign_suggest(pos_max, neg_min,
                                                                                          sample_num, precision)

s = time.time()
print('g assign {} h assign {}'.format(g_assign_bit, h_assign_bit))

exponent = FixedPointNumber.encode(0, g_modulo, g_max_int, precision).exponent
pack_time_s = time.time()
pack_g_h = []
mul = pow(FixedPointNumber.BASE, exponent)

for g_, h_ in zip(g, h):
    pack_g_h.append(pack((g_, h_), mul, g_modulo, h_modulo, offset=h_assign_bit))
    # pack_g_h.append(pack((g_, h_), g_modulo, g_max_int, h_modulo, h_max_int, h_assign_bit))

pack_time_e = time.time()
print('pack time', pack_time_e - pack_time_s)
en_paillier = [raw_encrypt(i, encrypter, exponent) for i in pack_g_h]
en_test = en_paillier[0]
for i in en_paillier[1:500]:
    en_test += i
en_test2 = en_paillier[500]
for i in en_paillier[501:]:
    en_test2 += i

g_sum_1, g_sum_2 = np.sum(g[0:500]), np.sum(g[500:])
h_sum_1, h_sum_2 = np.sum(h[0:500]), np.sum(h[500:])

print(g_sum_1, h_sum_1)
print(g_sum_2, h_sum_2)
de_rs = raw_decrypt(en_test, encrypter)
test_g_1, test_h_1 = unpack(de_rs, h_assign_bit, g_modulo, g_max_int, h_modulo, h_max_int, exponent)
test_g_1 = test_g_1 - 500*G_OFFSET
print(test_g_1, test_h_1)

split_info_1 = SplitInfo(sum_grad=en_test, sample_count=500)
split_info_2 = SplitInfo(sum_grad=en_test2, sample_count=500)
pack = SplitInfoPackage(h_assign_bit + g_assign_bit, capacity, 0)
pack.add(split_info_1)
pack.add(split_info_2)


# de_rs = raw_decrypt(en_test, encrypter)
#
# print(unpack(de_rs, h_assign_bit, g_modulo, g_max_int, h_modulo, h_max_int, exponent))
# e = time.time()
# print('take time {}'.format(e - s))
#
# s = time.time()
# en_g = [encrypter.encrypt(i) for i in g]
# en_h = [encrypter.encrypt(i) for i in h]
# g_rs = en_g[0]
# for i in en_g[1:]:
#     g_rs += i
# h_rs = en_h[0]
# for i in en_h[1:]:
#     h_rs += i
#
# de_g = encrypter.decrypt(g_rs)
# de_h = encrypter.decrypt(h_rs)
# print(de_g)
# print(de_h)
# e = time.time()
# print('take time {}'.format(e - s))
#
# print(g.sum())
# print(h.sum())