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

# Implement model of "Very deep convolutional networks for text classification"
# which can be seen at "http://www.aclweb.org/anthology/E17-1104"

import torch

import numpy as np

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier


class TextVDCNN(Classifier):
    def __init__(self, dataset, config):
        """all convolutional blocks
        4 kinds of conv blocks, which #feature_map are 64,128,256,512
        Depth:             9  17 29 49
        ------------------------------
        conv block 512:    2  4  4  6
        conv block 256:    2  4  4  10
        conv block 128:    2  4  10 16
        conv block 64:     2  4  10 16
        First conv. layer: 1  1  1  1
        """
        super(TextVDCNN, self).__init__(dataset, config)

        self.vdcnn_num_convs = {}
        self.vdcnn_num_convs[9] = [2, 2, 2, 2]
        self.vdcnn_num_convs[17] = [4, 4, 4, 4]
        self.vdcnn_num_convs[29] = [10, 10, 4, 4]
        self.vdcnn_num_convs[49] = [16, 16, 10, 6]
        self.num_kernels = [64, 128, 256, 512]

        self.vdcnn_depth = config.TextVDCNN.vdcnn_depth
        self.first_conv = torch.nn.Conv1d(config.embedding.dimension, 64, 3,
                                          padding=2)
        last_num_kernel = 64
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i, num_kernel in enumerate(self.num_kernels):
            tmp_convs = torch.nn.ModuleList()
            tmp_batch_norms = torch.nn.ModuleList()
            for _ in range(0, self.vdcnn_num_convs[self.vdcnn_depth][i]):
                tmp_convs.append(
                    torch.nn.Conv1d(last_num_kernel, num_kernel, 3, padding=2))
                tmp_batch_norms.append(torch.nn.BatchNorm1d(num_kernel))
            last_num_kernel = num_kernel
            self.convs.append(tmp_convs)
            self.batch_norms.append(tmp_batch_norms)

        self.top_k = self.config.TextVDCNN.top_k_max_pooling
        hidden_size = self.num_kernels[-1] * self.top_k
        self.linear1 = torch.nn.Linear(hidden_size, 2048)
        self.linear2 = torch.nn.Linear(2048, 2048)
        self.linear = torch.nn.Linear(2048, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.char_embedding.parameters()})
        params.append({'params': self.first_conv.parameters()})
        for i in range(0, len(self.num_kernels)):
            params.append({'params': self.convs[i].parameters()})
            params.append({'params': self.batch_norms[i].parameters()})
        params.append({'params': self.linear1.parameters()})
        params.append({'params': self.linear2.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):
        def convolutional_block(inputs, num_layers, convs, batch_norms):
            """Convolutional Block of VDCNN
            Convolutional block contains 2 conv layers, and can be repeated
            Temp Conv-->Batch Norm-->ReLU-->Temp Conv-->Batch Norm-->ReLU
            """
            hidden_layer = inputs
            for i in range(0, num_layers):
                batch_norm = batch_norms[i](convs[i](inputs))
                hidden_layer = torch.nn.functional.relu(batch_norm)
            return hidden_layer

        if self.config.feature.feature_names[0] == "token":
            embedding = self.token_embedding(
                batch[cDataset.DOC_TOKEN].to(self.config.device))
        else:
            embedding = self.char_embedding(
                batch[cDataset.DOC_CHAR].to(self.config.device))
        embedding = embedding.transpose(1, 2)

        # first conv layer (kernel_size=3, #feature_map=64)
        first_conv = self.first_conv(embedding)
        first_conv = torch.nn.functional.relu(first_conv)

        # all convolutional blocks
        conv_block = first_conv
        for i in range(0, len(self.num_kernels)):
            conv_block = convolutional_block(
                conv_block,
                num_layers=self.vdcnn_num_convs[self.vdcnn_depth][i],
                convs=self.convs[i],
                batch_norms=self.batch_norms[i])
            if i < len(self.num_kernels) - 1:
                # max-pooling with stride=2
                pool = torch.nn.functional.max_pool1d(conv_block,
                                                      kernel_size=3, stride=2)
            else:
                # k-max-pooling
                pool = torch.topk(conv_block, self.top_k)[0].view(
                    conv_block.size(0), -1)

        pool_shape = int(np.prod(pool.size()[1:]))
        doc_embedding = torch.reshape(pool, (-1, pool_shape))
        fc1 = self.linear1(doc_embedding)
        fc2 = self.linear2(fc1)
        return self.dropout(self.linear(fc2))
