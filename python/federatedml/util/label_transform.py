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

import copy

from fate_flow.entity.metric import Metric, MetricMeta
from federatedml.feature.instance import Instance
from federatedml.model_base import ModelBase
from federatedml.param.label_transform_param import LabelTransformParam
from federatedml.statistic.data_overview import get_label_count
from federatedml.util import consts, LOGGER


class LabelTransformer(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = LabelTransformParam()
        self.metric_name = "label_transform"
        self.metric_namespace = "train"
        self.metric_type = "LABEL_TRANSFORM"
        self.weight_mode = None

    def _init_model(self, params):
        self.model_param = params
        self.label_encoder = params.label_encoder
        self.need_run = params.need_run

    @staticmethod
    def replace_instance_label(instance, label_encoder):
        new_instance = copy.deepcopy(instance)
        new_instance.label = label_encoder[instance.label]
        return new_instance

    @staticmethod
    def replace_predict_label(predict_result, label_encoder):
        #@todo: replace label in predcit result
        pass

    @staticmethod
    def get_label_encoder(data, label_encoder):
        if label_encoder is not None:
            return label_encoder
        # @TODO: get label encoder dict
        # normal data instance
        if isinstance(data.first()[1], Instance):
            label_count = get_label_count(data)
            label_encoder = dict(zip(label_count.keys(), range(len(label_count.keys()))))
        # predict result
        else:
            pass
        return label_encoder

    def export_model(self):
        pass

    def load_model(self):
        pass

    def callback_info(self):
        metric_meta = MetricMeta(name='train',
                                 metric_type=self.metric_type,
                                 extra_metas={
                                     "label_encoder": self.label_encoder
                                 })

        self.callback_metric(metric_name=self.metric_name,
                             metric_namespace=self.metric_namespace,
                             metric_data=[Metric(self.metric_name, 0)])
        self.tracker.set_metric_meta(metric_namespace=self.metric_namespace,
                                     metric_name=self.metric_name,
                                     metric_meta=metric_meta)

    @staticmethod
    def transform_data_label(data, label_encoder):
        if isinstance(data.first()[1], Instance):
            return data.mapValues(lambda v: LabelTransformer.replace_instance_label(v, label_encoder))
        else:
            return data.mapValues(lambda v: LabelTransformer.replace_predict_label(v, label_encoder))

    def transform(self, data):
        LOGGER.info(f"Enter Label Transformer Transform")
        if self.label_encoder is None:
            raise ValueError(f"Input Label Encoder is None. Label Transform aborted.")

        label_encoder = self.label_encoder
        # revert label encoding if predict result
        if not isinstance(data.first()[1], Instance):
            label_encoder = dict(zip(self.label_encoder.values(), self.label_encoder.keys()))

        result_data = LabelTransformer.transform_data_label(data, label_encoder)
        result_data.schema = data.schema
        self.callback_info()

        return result_data

    def fit(self, data):
        LOGGER.info(f"Enter Label Transform Fit")

        if self.label_encoder is None:
            self.label_encoder = LabelTransformer.get_label_encoder(data, self.label_encoder)
        else:
            LOGGER.info(f"Label encoder provided.")

        result_data = LabelTransformer.transform_data_label(data, self.label_encoder)
        result_data.schema = data.schema
        self.callback_info()

        return result_data
