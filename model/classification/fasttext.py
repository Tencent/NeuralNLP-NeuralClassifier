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
from model.model_util import ActivationType


class FastText(torch.nn.Module):
    """Implement fasttext classification method
    Reference: "Bag of Tricks for Efficient Text Classification"
    """

    def __init__(self, dataset, config):
        super(FastText, self).__init__()
        self.config = config
        assert "token" in self.config.feature.feature_names
        self.token_embedding = \
            Embedding(dataset.token_map,
                      config.embedding.dimension,
                      cDataset.DOC_TOKEN, config,
                      padding_idx=dataset.VOCAB_PADDING,
                      pretrained_embedding_file=
                      config.feature.token_pretrained_file,
                      mode=EmbeddingProcessType.SUM, dropout=0,
                      init_type=config.embedding.initializer,
                      low=-config.embedding.uniform_bound,
                      high=config.embedding.uniform_bound,
                      std=config.embedding.random_stddev,
                      activation_type=ActivationType.NONE)
        if self.config.feature.token_ngram > 1:
            self.token_ngram_embedding = \
                Embedding(dataset.token_ngram_map,
                          config.embedding.dimension,
                          cDataset.DOC_TOKEN_NGRAM, config, 
                          padding_idx=dataset.VOCAB_PADDING,
                          mode=EmbeddingProcessType.SUM, dropout=0,
                          init_type=config.embedding.initializer,
                          low=-config.embedding.uniform_bound,
                          high=config.embedding.uniform_bound,
                          std=config.embedding.random_stddev,
                          activation_type=ActivationType.NONE)
        if "keyword" in self.config.feature.feature_names:
            self.keyword_embedding = \
                Embedding(dataset.keyword_map,
                          config.embedding.dimension,
                          cDataset.DOC_KEYWORD, config, 
                          padding_idx=dataset.VOCAB_PADDING,
                          pretrained_embedding_file=
                          config.feature.keyword_pretrained_file,
                          mode=EmbeddingProcessType.SUM, dropout=0,
                          init_type=config.embedding.initializer,
                          low=-config.embedding.uniform_bound,
                          high=config.embedding.uniform_bound,
                          std=config.embedding.random_stddev,
                          activation_type=ActivationType.NONE)
        if "topic" in self.config.feature.feature_names:
            self.topic_embedding = \
                Embedding(dataset.topic_map,
                          config.embedding.dimension,
                          cDataset.DOC_TOPIC, config, 
                          padding_idx=dataset.VOCAB_PADDING,
                          mode=EmbeddingProcessType.SUM, dropout=0,
                          init_type=config.embedding.initializer,
                          low=-config.embedding.uniform_bound,
                          high=config.embedding.uniform_bound,
                          std=config.embedding.random_stddev,
                          activation_type=ActivationType.NONE)
        self.linear = torch.nn.Linear(
            config.embedding.dimension, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        if self.config.feature.token_ngram > 1:
            params.append({'params': self.token_ngram_embedding.parameters()})
        if "keyword" in self.config.feature.feature_names:
            params.append({'params': self.keyword_embedding.parameters()})
        if "topic" in self.config.feature.feature_names:
            params.append({'params': self.topic_embedding.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        """Update lr
        """
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0

    def forward(self, batch):
        doc_embedding = self.token_embedding(
            batch[cDataset.DOC_TOKEN].to(self.config.device),
            batch[cDataset.DOC_TOKEN_OFFSET].to(self.config.device))
        length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
        if self.config.feature.token_ngram > 1:
            doc_embedding += self.token_ngram_embedding(
                batch[cDataset.DOC_TOKEN_NGRAM].to(self.config.device),
                batch[cDataset.DOC_TOKEN_NGRAM_OFFSET].to(self.config.device))
            length += batch[cDataset.DOC_TOKEN_NGRAM_LEN].to(self.config.device)
        if "keyword" in self.config.feature.feature_names:
            doc_embedding += self.keyword_embedding(
                batch[cDataset.DOC_KEYWORD].to(self.config.device),
                batch[cDataset.DOC_KEYWORD_OFFSET].to(self.config.device))
            length += batch[cDataset.DOC_KEYWORD_LEN].to(self.config.device)
        if "topic" in self.config.feature.feature_names:
            doc_embedding += self.topic_embedding(
                batch[cDataset.DOC_TOPIC].to(self.config.device),
                batch[cDataset.DOC_TOPIC_OFFSET].to(self.config.device))
            length += batch[cDataset.DOC_TOPIC_LEN].to(self.config.device)

        doc_embedding /= length.resize_(doc_embedding.size()[0], 1)
        doc_embedding = self.dropout(doc_embedding)
        return self.linear(doc_embedding)
