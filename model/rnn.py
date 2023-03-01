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

import torch

from util import Type


class RNNType(Type):
    RNN = 'RNN'
    LSTM = 'LSTM'
    GRU = 'GRU'

    @classmethod
    def str(cls):
        return ",".join([cls.RNN, cls.LSTM, cls.GRU])


class RNN(torch.nn.Module):
    """
    One layer rnn.
    """

    def __init__(self, input_size, hidden_size, num_layers=1,
                 nonlinearity="tanh", bias=True, batch_first=False, dropout=0,
                 bidirectional=False, rnn_type=RNNType.GRU):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        if rnn_type == RNNType.LSTM:
            self.rnn = torch.nn.LSTM(
                input_size, hidden_size, num_layers=num_layers, bias=bias,
                batch_first=batch_first, dropout=dropout,
                bidirectional=bidirectional)
        elif rnn_type == RNNType.GRU:
            self.rnn = torch.nn.GRU(
                input_size, hidden_size, num_layers=num_layers, bias=bias,
                batch_first=batch_first, dropout=dropout,
                bidirectional=bidirectional)
        elif rnn_type == RNNType.RNN:
            self.rnn = torch.nn.RNN(
                input_size, hidden_size, vnonlinearity=nonlinearity, bias=bias,
                batch_first=batch_first, dropout=dropout,
                bidirectional=bidirectional)
        else:
            raise TypeError(
                "Unsupported rnn init type: %s. Supported rnn type is: %s" % (
                    rnn_type, RNNType.str()))

    def forward(self, inputs, seq_lengths=None, init_state=None,
                ori_state=False):
        """
        Args:
            inputs:
            seq_lengths:
            init_state:
            ori_state: If true, will return ori state generate by rnn. Else will
                       will return formatted state
        :return:
        """
        if seq_lengths is not None:
            seq_lengths = seq_lengths.int()
            sorted_seq_lengths, indices = torch.sort(seq_lengths,
                                                     descending=True)
            if self.batch_first:
                sorted_inputs = inputs[indices]
            else:
                sorted_inputs = inputs[:, indices]
            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_seq_lengths.cpu(), batch_first=self.batch_first)
            outputs, state = self.rnn(packed_inputs, init_state)
        else:
            outputs, state = self.rnn(inputs, init_state)

        if ori_state:
            return outputs, state
        if self.rnn_type == RNNType.LSTM:
            state = state[0]
        if self.bidirectional:
            last_layers_hn = state[2 * (self.num_layers - 1):]
            last_layers_hn = torch.cat(
                (last_layers_hn[0], last_layers_hn[1]), 1)
        else:
            last_layers_hn = state[self.num_layers - 1:]
            last_layers_hn = last_layers_hn[0]

        _, revert_indices = torch.sort(indices, descending=False)
        last_layers_hn = last_layers_hn[revert_indices]
        pad_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=self.batch_first)
        if self.batch_first:
            pad_output = pad_output[revert_indices]
        else:
            pad_output = pad_output[:, revert_indices]
        return pad_output, last_layers_hn
