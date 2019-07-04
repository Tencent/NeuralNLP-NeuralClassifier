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
import torch.nn.functional as F

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.rnn import RNN


class TextRCNN(Classifier):
    """TextRNN + TextCNN
    """
    def __init__(self, dataset, config):
        super(TextRCNN, self).__init__(dataset, config)
        self.rnn = RNN(
            config.embedding.dimension, config.TextRCNN.hidden_dimension,
            num_layers=config.TextRCNN.num_layers,
            batch_first=True, bidirectional=config.TextRCNN.bidirectional,
            rnn_type=config.TextRCNN.rnn_type)

        hidden_dimension = config.TextRCNN.hidden_dimension
        if config.TextRCNN.bidirectional:
            hidden_dimension *= 2
        self.kernel_sizes = config.TextRCNN.kernel_sizes
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(
                hidden_dimension, config.TextRCNN.num_kernels,
                kernel_size, padding=kernel_size - 1))

        self.top_k = self.config.TextRCNN.top_k_max_pooling
        hidden_size = len(config.TextRCNN.kernel_sizes) * \
                      config.TextRCNN.num_kernels * self.top_k

        self.linear = torch.nn.Linear(hidden_size, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.char_embedding.parameters()})
        params.append({'params': self.rnn.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):
        if self.config.feature.feature_names[0] == "token":
            embedding = self.token_embedding(
                batch[cDataset.DOC_TOKEN].to(self.config.device))
            seq_length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
        else:
            embedding = self.char_embedding(
                batch[cDataset.DOC_CHAR].to(self.config.device))
            seq_length = batch[cDataset.DOC_CHAR_LEN].to(self.config.device)
        output, _ = self.rnn(embedding, seq_length)

        doc_embedding = output.transpose(1, 2)
        pooled_outputs = []
        for _, conv in enumerate(self.convs):
            convolution = F.relu(conv(doc_embedding))
            pooled = torch.topk(convolution, self.top_k)[0].view(
                convolution.size(0), -1)
            pooled_outputs.append(pooled)

        doc_embedding = torch.cat(pooled_outputs, 1)

        return self.dropout(self.linear(doc_embedding))
