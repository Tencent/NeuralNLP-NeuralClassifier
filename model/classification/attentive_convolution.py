#!usr/bin/env python
# coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import torch

from model.classification.classifier import Classifier
from model.layers import AdditiveAttention2D
from model.layers import DotProductAttention
from model.layers import Highway
from model.model_util import init_tensor
from util import Type


class AttentiveConvNetType(Type):
    LIGHT = "light"
    ADVANCED = "advanced"

    @classmethod
    def str(cls):
        return ",".join(cls.LIGHT, cls.ADVANCED)


class AttentionType(Type):
    DOT = "dot"
    BILINEAR = "bilinear"
    ADDITIVE_PROJECTION = "additive_projection"

    @classmethod
    def str(cls):
        return ",".join(cls.DOT, cls.BILINEAR, cls.ADDITIVE_PROJECTION)


class AttentiveConvNet(Classifier):
    """Attentive Convolution:
    Equipping CNNs with RNN-style Attention Mechanisms
    """

    def __init__(self, dataset, config):
        super(AttentiveConvNet, self).__init__(dataset, config)
        self.attentive_conv_net_type = config.AttentiveConvNet.type
        self.attention_type = config.AttentiveConvNet.attention_type
        self.dim = config.embedding.dimension
        self.attention_dim = self.dim
        self.margin_size = config.AttentiveConvNet.margin_size
        assert self.margin_size % 2 == 1, \
            "AttentiveConvNet margin size should be odd!"

        self.radius = int(self.margin_size / 2)
        if self.attentive_conv_net_type == AttentiveConvNetType.ADVANCED:
            self.attention_dim *= 2
            self.x_context_highway = self.get_highway(self.dim,
                                                      self.margin_size)

            self.x_self_highway = self.get_highway(self.dim, 1)

            self.a_context_highway = self.get_highway(self.dim,
                                                      self.margin_size)
            self.a_self_highway = self.get_highway(self.dim, 1)
            self.beneficiary_highway = self.get_highway(self.dim, 1)

        if self.attention_type == AttentionType.DOT:
            self.dot_product_attention = DotProductAttention(1.0)
        elif self.attention_type == AttentionType.BILINEAR:
            self.bilinear_matrix = init_tensor(
                torch.empty(self.attention_dim, self.attention_dim)).to(
                config.device)
            self.dot_product_attention = DotProductAttention(1.0)
        elif self.attention_type == AttentionType.ADDITIVE_PROJECTION:
            self.additive_projection = AdditiveAttention2D(self.attention_dim)
        else:
            raise TypeError(
                "Unsupported AttentionType: %s." % self.attention_type)

        self.attentive_conv = init_tensor(
            torch.empty(self.attention_dim, self.dim)).to(config.device)
        self.x_conv = torch.nn.Sequential(
            torch.nn.Conv1d(self.dim, self.dim, self.margin_size,
                            padding=self.radius),
            torch.nn.Tanh())
        self.bias = torch.zeros([self.dim]).to(config.device)
        self.hidden_size = config.AttentiveConvNet.hidden_size
        self.hidden1_matrix = init_tensor(
            torch.empty(self.dim, self.hidden_size)).to(config.device)
        self.hidden2_matrix = init_tensor(
            torch.empty(self.hidden_size, self.hidden_size)).to(config.device)
        self.linear = torch.nn.Linear(self.dim + 2 * self.hidden_size,
                                      len(dataset.label_map))

    @staticmethod
    def get_highway(dimension, margin_size):
        radius = int(margin_size / 2)
        transformer_gate = torch.nn.Sequential(
            torch.nn.Conv1d(dimension, dimension, margin_size, padding=radius),
            torch.nn.Sigmoid())
        transformer_forward = torch.nn.Sequential(
            torch.nn.Conv1d(dimension, dimension, margin_size, padding=radius),
            torch.nn.Tanh())
        return Highway(transformer_gate, transformer_forward)

    def get_parameter_optimizer_dict(self):
        params = super(AttentiveConvNet,
                       self).get_parameter_optimizer_dict()
        if self.attentive_conv_net_type == AttentiveConvNetType.ADVANCED:
            params.append({'params': self.x_context_highway.parameters()})
            params.append({'params': self.x_self_highway.parameters()})
            params.append({'params': self.a_context_highway.parameters()})
            params.append({'params': self.a_self_highway.parameters()})
            params.append({'params': self.beneficiary_highway.parameters()})
        if self.attention_type == AttentionType.DOT:
            params.append({'params': self.dot_product_attention.parameters()})
        elif self.attention_type == AttentionType.BILINEAR:
            params.append({'params': self.bilinear_matrix})
            params.append({'params': self.dot_product_attention.parameters()})
        elif self.attention_type == AttentionType.ADDITIVE_PROJECTION:
            params.append({'params': self.additive_projection.parameters()})

        params.append({'params': self.attentive_conv})
        params.append({'params': self.x_conv.parameters()})
        params.append({'params': self.hidden1_matrix})
        params.append({'params': self.hidden2_matrix})
        params.append({'params': self.linear.parameters()})

        return params

    def forward(self, batch):

        embedding, _, _ = self.get_embedding(batch)
        if self.attentive_conv_net_type == AttentiveConvNetType.LIGHT:
            x_multi_granularity, a_multi_granularity, x_beneficiary = \
                embedding, embedding, embedding
        elif self.attentive_conv_net_type == AttentiveConvNetType.ADVANCED:
            embedding = embedding.permute(0, 2, 1)
            source_context = self.x_context_highway(embedding)
            source_self = self.x_self_highway(embedding)
            x_multi_granularity = \
                torch.cat([source_context, source_self], 1).permute(0, 2, 1)

            focus_context = self.a_context_highway(embedding)
            focus_self = self.a_self_highway(embedding)
            a_multi_granularity = \
                torch.cat([focus_context, focus_self], 1).permute(0, 2, 1)

            x_beneficiary = self.beneficiary_highway(
                embedding).permute(0, 2, 1)
        else:
            raise TypeError(
                "Unsupported AttentiveConvNetType: %s." %
                self.attentive_conv_net_type)

        if self.attention_type == AttentionType.DOT:
            attentive_context = self.dot_product_attention(
                x_multi_granularity, a_multi_granularity, a_multi_granularity)
        elif self.attention_type == AttentionType.BILINEAR:
            x_trans = x_multi_granularity.matmul(self.bilinear_matrix)
            attentive_context = self.dot_product_attention(
                x_trans, a_multi_granularity, a_multi_granularity)
        elif self.attention_type == AttentionType.ADDITIVE_PROJECTION:
            attentive_context = self.additive_projection(
                a_multi_granularity, x_multi_granularity)

        attentive_conv = attentive_context.matmul(self.attentive_conv)
        x_conv = self.x_conv(x_beneficiary.permute(0, 2, 1)).permute(0, 2, 1)
        attentive_convolution = \
            torch.tanh(attentive_conv + x_conv + self.bias).permute(0, 2, 1)
        hidden = torch.nn.functional.max_pool1d(
            attentive_convolution,
            kernel_size=attentive_convolution.size()[-1]).squeeze()
        hidden1 = hidden.matmul(self.hidden1_matrix)
        hidden2 = hidden1.matmul(self.hidden2_matrix)
        hidden_layer = torch.cat([hidden, hidden1, hidden2], 1)

        return self.dropout(self.linear(hidden_layer))
