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

from pipeline.param.base_param import BaseParam


class ModelInterpretParam(BaseParam):

    def __init__(self, need_explain=False, method='shap', interpret_limit=10, shap_subset_sample_num='auto'
                 ,reference_type='all_zero', random_seed=100, ):
        super(ModelInterpretParam, self).__init__()
        self.need_explain = need_explain
        self.method = method
        self.interpret_limit = interpret_limit
        self.shap_subset_sample_num = shap_subset_sample_num
        self.random_seed = random_seed
        self.reference_type = reference_type

    def check(self):
        self.check_boolean(self.need_explain, 'need_explain')
        self.check_positive_integer(self.interpret_limit, 'interpret_limit')
        self.check_positive_integer(self.random_seed, 'random_seed')
        if self.shap_subset_sample_num != 'auto':
            self.check_positive_integer(self.shap_subset_sample_num, 'shap_subset_sample_num')


