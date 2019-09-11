#!/usr/bin/env python
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

import math

import torch

from model.model_util import init_tensor


class SumAttention(torch.nn.Module):
    """
    Reference: Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, input_dimension, attention_dimension, device, dropout=0):
        super(SumAttention, self).__init__()
        self.attention_matrix = \
            torch.nn.Linear(input_dimension, attention_dimension).to(device)
        self.attention_vector = torch.nn.Linear(attention_dimension, 1, bias=False).to(device)
        init_tensor(self.attention_matrix.weight)
        init_tensor(self.attention_vector.weight)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, inputs):
        if inputs.size(1) == 1:
            return self.dropout(inputs.squeeze())
        u = torch.tanh(self.attention_matrix(inputs))
        v = self.attention_vector(u)
        alpha = torch.nn.functional.softmax(v, 1).squeeze().unsqueeze(1)
        return self.dropout(torch.matmul(alpha, inputs).squeeze())


class AdditiveAttention(torch.nn.Module):
    """Also known as Soft Attention or Bahdanau Attention
    Reference:
        Neural machine translation by jointly learning to align and translate
    """

    def __init__(self, dim, dropout=0):
        super(AdditiveAttention, self).__init__()
        self.w_attention_matrix = init_tensor(torch.empty(dim, dim))
        self.u_attention_matrix = init_tensor(torch.empty(dim, dim))
        self.v_attention_vector = init_tensor(torch.empty(dim, 1))

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, s, h):
        raise NotImplementedError


class AdditiveAttention1D(AdditiveAttention):
    """
    Input shape is: [batch, dim] and [batch, seq_len, dim]
    Output is same with the first input
    """

    def forward(self, s, h):
        s_attention = s.matmul(self.w_attention_matrix).unsqueeze(1)
        h_attention = h.matmul(self.u_attention_matrix)
        attention = torch.tanh(s_attention + h_attention)
        attention = attention.matmul(self.v_attention_vector).squeeze()
        attention_weight = torch.nn.functional.softmax(attention, -1)
        return self.dropout(attention_weight.unsqueeze(1).matmul(h).squeeze())


class AdditiveAttention2D(AdditiveAttention):
    """
    Input shape is: [batch, seq_len, dim] and [batch, seq_len, dim]
    Output is same with the first input
    """

    def forward(self, s, h):
        s_attention = s.matmul(self.w_attention_matrix).unsqueeze(2)
        h_attention = h.matmul(self.u_attention_matrix).unsqueeze(1)
        seq_len = h.size(1)
        h_attention = h_attention.expand(-1, seq_len, -1, -1)
        attention = torch.nn.functional.tanh(s_attention + h_attention)
        attention = attention.matmul(self.v_attention_vector).squeeze()
        attention_weight = torch.nn.functional.softmax(attention, -1)
        return self.dropout(attention_weight.unsqueeze(2).matmul(h).squeeze())


class DotProductAttention(torch.nn.Module):
    """
    Reference: Attention is all you need
    Input shape is: [batch, seq_len, dim_k] and [batch, seq_len, dim_k]
                    [batch, seq_len, dim_v]
    Output is same with the third input
    """

    def __init__(self, scaling_factor=None, dropout=0):
        super(DotProductAttention, self).__init__()
        self.scaling_factor = scaling_factor
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, q, k, v):
        if self.scaling_factor is None:
            self.scaling_factor = 1 / math.sqrt(q.size(2))
        e = q.matmul(k.permute(0, 2, 1)) / self.scaling_factor
        attention_weight = torch.nn.functional.softmax(e, -1)
        return self.dropout(attention_weight.matmul(v))


class MultiHeadAttention(torch.nn.Module):
    """
    Reference: Attention is all you need
    """

    def __init__(self, dimension, dk, dv, head_number,
                 scaling_factor, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.dk = dk
        self.dv = dv
        self.head_number = head_number
        self.q_linear = torch.nn.Linear(dimension, head_number * dk)
        self.k_linear = torch.nn.Linear(dimension, head_number * dk)
        self.v_linear = torch.nn.Linear(dimension, head_number * dv)
        self.scaling_factor = scaling_factor
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, q, k, v):
        def _reshape_permute(x, d, head_number):
            x = x.view(x.size(0), x.size(1), head_number, d)
            return x.permute(0, 2, 1, 3)

        q_trans = _reshape_permute(self.q_linear(q), self.dk, self.head_number)
        k_trans = _reshape_permute(self.k_linear(k), self.dk, self.head_number)
        v_trans = _reshape_permute(self.v_linear(v), self.dv, self.head_number)

        e = q_trans.matmul(k_trans.permute(0, 1, 3, 2)) / self.scaling_factor
        attention_weight = torch.nn.functional.softmax(e, -1)
        output = attention_weight.matmul(v_trans).permute(0, 2, 1, 3)
        output = output.view(output.size(0), output.size(1),
                             output.size(2) * output.size(3))
        return self.dropout(output)


class Highway(torch.nn.Module):
    """
    Reference: Highway Networks.
    For now we don't limit the type of the gate and forward.
    Caller should init Highway with transformer and carry and guarantee the dim
    to be matching.
    """

    def __init__(self, transformer_gate, transformer_forward):
        super(Highway, self).__init__()
        self.transformer_forward = transformer_forward
        self.transformer_gate = transformer_gate

    def forward(self, x, gate_input=None, forward_input=None):
        if gate_input is None:
            gate_input = x
        if forward_input is None:
            forward_input = x
        gate = self.transformer_gate(gate_input)
        forward = self.transformer_forward(forward_input)
        return gate * forward + (1 - gate) * x
