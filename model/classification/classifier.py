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

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.embedding import Embedding
from model.embedding import EmbeddingProcessType
from model.embedding import EmbeddingType
from model.embedding import RegionEmbeddingLayer
from model.model_util import ActivationType


class Classifier(torch.nn.Module):
    def __init__(self, dataset, config):
        super(Classifier, self).__init__()
        self.config = config
        assert len(self.config.feature.feature_names) == 1
        assert self.config.feature.feature_names[0] == "token" or \
               self.config.feature.feature_names[0] == "char"
        if config.embedding.type == EmbeddingType.EMBEDDING:
            self.token_embedding = \
                Embedding(dataset.token_map, config.embedding.dimension,
                          cDataset.DOC_TOKEN, config, dataset.VOCAB_PADDING,
                          pretrained_embedding_file=
                          config.feature.token_pretrained_file,
                          mode=EmbeddingProcessType.FLAT,
                          dropout=self.config.embedding.dropout,
                          init_type=self.config.embedding.initializer,
                          low=-self.config.embedding.uniform_bound,
                          high=self.config.embedding.uniform_bound,
                          std=self.config.embedding.random_stddev,
                          fan_mode=self.config.embedding.fan_mode,
                          activation_type=ActivationType.NONE,
                          model_mode=dataset.model_mode)
            self.char_embedding = \
                Embedding(dataset.char_map, config.embedding.dimension,
                          cDataset.DOC_CHAR, config, dataset.VOCAB_PADDING,
                          mode=EmbeddingProcessType.FLAT,
                          dropout=self.config.embedding.dropout,
                          init_type=self.config.embedding.initializer,
                          low=-self.config.embedding.uniform_bound,
                          high=self.config.embedding.uniform_bound,
                          std=self.config.embedding.random_stddev,
                          fan_mode=self.config.embedding.fan_mode,
                          activation_type=ActivationType.NONE,
                          model_mode=dataset.model_mode)
        elif config.embedding.type == EmbeddingType.REGION_EMBEDDING:
            self.token_embedding = RegionEmbeddingLayer(
                dataset.token_map, config.embedding.dimension,
                config.embedding.region_size, cDataset.DOC_TOKEN, config,
                padding=dataset.VOCAB_PADDING,
                pretrained_embedding_file=
                config.feature.token_pretrained_file,
                dropout=self.config.embedding.dropout,
                init_type=self.config.embedding.initializer,
                low=-self.config.embedding.uniform_bound,
                high=self.config.embedding.uniform_bound,
                std=self.config.embedding.random_stddev,
                fan_mode=self.config.embedding.fan_mode,
                model_mode=dataset.model_mode,
                region_embedding_type=config.embedding.region_embedding_type)

            self.char_embedding = RegionEmbeddingLayer(
                dataset.char_map, config.embedding.dimension,
                config.embedding.region_size, cDataset.DOC_CHAR, config,
                padding=dataset.VOCAB_PADDING,
                dropout=self.config.embedding.dropout,
                init_type=self.config.embedding.initializer,
                low=-self.config.embedding.uniform_bound,
                high=self.config.embedding.uniform_bound,
                std=self.config.embedding.random_stddev,
                fan_mode=self.config.embedding.fan_mode,
                model_mode=dataset.model_mode,
                region_embedding_type=config.embedding.region_embedding_type)
        else:
            raise TypeError(
                "Unsupported embedding type: %s. " % config.embedding.type)
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_embedding(self, batch, pad_shape=None, pad_value=0):
        if self.config.feature.feature_names[0] == "token":
            token_id = batch[cDataset.DOC_TOKEN].to(self.config.device)
            if pad_shape is not None:
                token_id = torch.nn.functional.pad(
                    token_id, pad_shape, mode='constant', value=pad_value)
            embedding = self.token_embedding(token_id)
            length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
            mask = batch[cDataset.DOC_TOKEN_MASK].to(self.config.device)
        else:
            char_id = batch[cDataset.DOC_TOKEN].to(self.config.device)
            if pad_shape is not None:
                char_id = torch.nn.functional.pad(
                    char_id, pad_shape, mode='constant', value=pad_value)
            embedding = self.token_embedding(char_id)
            length = batch[cDataset.DOC_CHAR_LEN].to(self.config.device)
            mask = batch[cDataset.DOC_CHAR_MASK].to(self.config.device)
        return embedding, length, mask

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append(
            {'params': self.token_embedding.parameters(), 'is_embedding': True})
        params.append(
            {'params': self.char_embedding.parameters(), 'is_embedding': True})
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
        raise NotImplementedError
