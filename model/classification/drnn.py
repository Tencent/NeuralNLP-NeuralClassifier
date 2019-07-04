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

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.rnn import RNN
from model.rnn import RNNType


class DRNN(Classifier):
    def __init__(self, dataset, config):
        super(DRNN, self).__init__(dataset, config)
        self.rnn_type = config.DRNN.rnn_type
        self.forward_rnn = RNN(
            config.embedding.dimension, config.DRNN.hidden_dimension,
            batch_first=True, rnn_type=config.DRNN.rnn_type)
        if config.DRNN.bidirectional:
            self.backward_rnn = RNN(
                config.embedding.dimension, config.DRNN.hidden_dimension,
                batch_first=True, rnn_type=config.DRNN.rnn_type)
        self.window_size = config.DRNN.window_size
        self.dropout = torch.nn.Dropout(p=config.DRNN.cell_hidden_dropout)
        self.hidden_dimension = config.DRNN.hidden_dimension
        if config.DRNN.bidirectional:
            self.hidden_dimension *= 2
        self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dimension)

        self.mlp = torch.nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.linear = torch.nn.Linear(self.hidden_dimension,
                                      len(dataset.label_map))

    def get_parameter_optimizer_dict(self):
        params = super(DRNN, self).get_parameter_optimizer_dict()
        params.append({'params': self.forward_rnn.parameters()})
        if self.config.DRNN.bidirectional:
            params.append({'params': self.backward_rnn.parameters()})
        params.append({'params': self.batch_norm.parameters()})
        params.append({'params': self.mlp.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def forward(self, batch):
        front_pad_embedding, _, mask = self.get_embedding(
            batch, [self.window_size - 1, 0], cDataset.VOCAB_PADDING_LEARNABLE)
        if self.config.DRNN.bidirectional:
            tail_pad_embedding, _, _ = self.get_embedding(
                batch, [0, self.window_size - 1],
                cDataset.VOCAB_PADDING_LEARNABLE)
        batch_size = front_pad_embedding.size(0)
        mask = mask.unsqueeze(2)

        front_slice_embedding_list = \
            [front_pad_embedding[:, i:i + self.window_size, :] for i in
             range(front_pad_embedding.size(1) - self.window_size + 1)]

        front_slice_embedding = torch.cat(front_slice_embedding_list, dim=0)

        state = None
        for i in range(front_slice_embedding.size(1)):
            _, state = self.forward_rnn(front_slice_embedding[:, i:i + 1, :],
                                        init_state=state, ori_state=True)
            if self.rnn_type == RNNType.LSTM:
                state[0] = self.dropout(state[0])
            else:
                state = self.dropout(state)
        front_state = state[0] if self.rnn_type == RNNType.LSTM else state
        front_state = front_state.transpose(0, 1)
        front_hidden = torch.cat(front_state.split(batch_size, dim=0), dim=1)
        front_hidden = front_hidden * mask

        hidden = front_hidden
        if self.config.DRNN.bidirectional:
            tail_slice_embedding_list = list()
            for i in range(tail_pad_embedding.size(1) - self.window_size + 1):
                slice_embedding = \
                    tail_pad_embedding[:, i:i + self.window_size, :]
                tail_slice_embedding_list.append(slice_embedding)
            tail_slice_embedding = torch.cat(tail_slice_embedding_list, dim=0)

            state = None
            for i in range(tail_slice_embedding.size(1), 0, -1):
                _, state = self.backward_rnn(
                    tail_slice_embedding[:, i - 1:i, :],
                    init_state=state, ori_state=True)
                if i != tail_slice_embedding.size(1) - 1:
                    if self.rnn_type == RNNType.LSTM:
                        state[0] = self.dropout(state[0])
                    else:
                        state = self.dropout(state)
            tail_state = state[0] if self.rnn_type == RNNType.LSTM else state
            tail_state = tail_state.transpose(0, 1)
            tail_hidden = torch.cat(tail_state.split(batch_size, dim=0), dim=1)
            tail_hidden = tail_hidden * mask
            hidden = torch.cat([hidden, tail_hidden], dim=2)

        hidden = hidden.transpose(1, 2).contiguous()

        batch_normed = self.batch_norm(hidden).transpose(1, 2)
        batch_normed = batch_normed * mask
        mlp_hidden = self.mlp(batch_normed)
        mlp_hidden = mlp_hidden * mask
        neg_mask = (mask - 1) * 65500.0
        mlp_hidden = mlp_hidden + neg_mask
        max_pooling = torch.nn.functional.max_pool1d(
            mlp_hidden.transpose(1, 2), mlp_hidden.size(1)).squeeze()
        return self.linear(self.dropout(max_pooling))
