#!/usr/bin/env python 
# coding: utf-8 
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
from model.rnn import RNN
from util import Type

from dataset.classification_dataset import ClassificationDataset as cDataset

class HMCN(Classifier):
    """ Implement HMCN(Hierarchical Multi-Label Classification Networks)
        Reference: "Hierarchical Multi-Label Classification Networks"
    """

    def __init__(self, dataset, config):
        super(HMCN, self).__init__(dataset, config)
        self.hierarchical_depth = config.HMCN.hierarchical_depth
        self.hierarchical_class = dataset.hierarchy_classes
        self.global2local = config.HMCN.global2local
        self.rnn = RNN(
            config.embedding.dimension, config.TextRNN.hidden_dimension, 
            num_layers=config.TextRNN.num_layers, batch_first=True,
            bidirectional=config.TextRNN.bidirectional,
            rnn_type=config.TextRNN.rnn_type)
        hidden_dimension = config.TextRNN.hidden_dimension
        if config.TextRNN.bidirectional:
            hidden_dimension *= 2
        
        self.local_layers = torch.nn.ModuleList()
        self.global_layers = torch.nn.ModuleList()
        for i in range(1, len(self.hierarchical_depth)):
            self.global_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dimension + self.hierarchical_depth[i-1], self.hierarchical_depth[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.hierarchical_depth[i]),
                    torch.nn.Dropout(p=0.5)
                ))
            self.local_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hierarchical_depth[i], self.global2local[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.global2local[i]),
                    torch.nn.Linear(self.global2local[i], self.hierarchical_class[i-1])
                ))

        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)
        self.linear = torch.nn.Linear(self.hierarchical_depth[-1], len(dataset.label_map))
        self.linear.apply(self._init_weight)
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)
        
    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1) 

    def get_parameter_optimizer_dict(self):
        params = super(HMCN, self).get_parameter_optimizer_dict() 
        params.append({'params': self.rnn.parameters()})
        params.append({'params': self.local_layers.parameters()})
        params.append({'params': self.global_layers.parameters()})
        params.append({'params': self.linear.parameters()})
        return params 

    def update_lr(self, optimizer, epoch):
        """ Update lr
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
            length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
        else:
            embedding = self.char_embedding(
                    batch[cDataset.DOC_TOKEN].to(self.config.device))
            length = batch[cDataset.DOC_CHAR_LEN].to(self.config.device)
        
        output, last_hidden = self.rnn(embedding, length)
        doc_embedding = torch.sum(output, 1) / length.unsqueeze(1) 
        local_layer_outputs = []
        global_layer_activation = doc_embedding
        batch_size = doc_embedding.size()[0]
        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            local_layer_activation = global_layer(global_layer_activation)
            local_layer_outputs.append(local_layer(local_layer_activation))
            if i < len(self.global_layers)-1:
                global_layer_activation = torch.cat((local_layer_activation, doc_embedding), 1)
            else:
                global_layer_activation = local_layer_activation

        global_layer_output = self.linear(global_layer_activation)
        local_layer_output = torch.cat(local_layer_outputs, 1)
        return global_layer_output, local_layer_output, 0.5 * global_layer_output + 0.5 * local_layer_output   
