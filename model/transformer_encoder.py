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

"""
Transformer Encoder:
    Heavily borrowed from https://github.com/jadore801120/attention-is-all-you-need-pytorch/
Star-Transformer Encode:
    https://arxiv.org/pdf/1902.09113v2.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class StarEncoderLayer(nn.Module):
    ''' Star-Transformer: https://arxiv.org/pdf/1902.09113v2.pdf '''

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super(StarEncoderLayer, self).__init__()
        self.slf_attn_satellite = MultiHeadAttention(
            n_head, d_model, d_k, d_v, use_star=True, dropout=dropout)
        self.slf_attn_relay = MultiHeadAttention(
            n_head, d_model, d_k, d_v, use_star=True, dropout=dropout)

    def forward(self, h, e, s, non_pad_mask=None, slf_attn_mask=None):
        # satellite node
        batch_size, seq_len, d_model = h.size()
        h_extand = torch.zeros(batch_size, seq_len+2, d_model, dtype=torch.float, device=h.device)
        h_extand[:, 1:seq_len+1, :] = h  # head and tail padding(not cycle)
        s = s.reshape([batch_size, 1, d_model])
        s_expand = s.expand([batch_size, seq_len, d_model])
        context = torch.cat((h_extand[:, 0:seq_len, :],
                             h_extand[:, 1:seq_len+1, :],
                             h_extand[:, 2:seq_len+2, :],
                             e,
                             s_expand),
                            2)
        context = context.reshape([batch_size*seq_len, 5, d_model])
        h = h.reshape([batch_size*seq_len, 1, d_model])

        h, _ = self.slf_attn_satellite(
            h, context, context, mask=slf_attn_mask)
        h = torch.squeeze(h, 1).reshape([batch_size, seq_len, d_model])
        if non_pad_mask is not None:
            h *= non_pad_mask

        # virtual relay node
        s_h = torch.cat((s, h), 1)
        s, _ = self.slf_attn_relay(
            s, s_h, s_h, mask=slf_attn_mask)
        s = torch.squeeze(s, 1)

        return h, s
