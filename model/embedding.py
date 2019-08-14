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

import numpy as np
import torch
import torch.nn as nn

from model.model_util import ActivationType
from model.model_util import FAN_MODE
from model.model_util import InitType
from model.model_util import init_tensor
from util import Logger
from util import Type, ModeType


class EmbeddingType(Type):
    """Standard names for embedding type
    The following keys are defined:
    * `EMBEDDING`: Return the embedding after lookup.
    * `REGION_EMBEDDING`: Return the region embedding.
        Reference: A New Method of Region Embedding for Text Classification
    """
    EMBEDDING = 'embedding'
    REGION_EMBEDDING = 'region_embedding'
    
    @classmethod
    def str(cls):
        return ",".join([cls.EMBEDDING, cls.REGION_EMBEDDING])


class EmbeddingProcessType(Type):
    """Standard names for embedding mode
    Given the vocab tensor shape[batch_size, sequence_len].
    The following keys are defined:
    * `FLAT`: Normal mode, return tensor shape will be
    *         [batch_size, sequence_len, embedding_size]
    * `MEAN`: Mean mode, return tensor shape will be
    *         [batch_size, embedding_size]
    * `SUM`: Sum mode, return tensor shape will be
    *        [batch_size, embedding_size]
    """
    FLAT = 'flat'
    MEAN = 'mean'
    SUM = 'sum'
    
    @classmethod
    def str(cls):
        return ",".join([cls.FLAT, cls.MEAN, cls.SUM])


class Embedding(torch.nn.Module):
    def __init__(self, dict_map, embedding_dim, name, config, padding_idx=None,
                 pretrained_embedding_file=None, mode=EmbeddingProcessType.FLAT,
                 dropout=0, init_type=InitType.XAVIER_UNIFORM, low=0, high=1,
                 mean=0, std=1, activation_type=ActivationType.NONE,
                 fan_mode=FAN_MODE.FAN_IN, negative_slope=0,
                 model_mode=ModeType.TRAIN):
        super(Embedding, self).__init__()
        self.logger = Logger(config)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.mode = mode
        if self.mode == EmbeddingProcessType.FLAT:
            self.embedding = torch.nn.Embedding(
                len(dict_map), embedding_dim, padding_idx=padding_idx)
        else:
            self.embedding = torch.nn.EmbeddingBag(
                len(dict_map), embedding_dim, mode=mode)
        embedding_lookup_table = init_tensor(
            tensor=torch.empty(len(dict_map), embedding_dim),
            init_type=init_type, low=low, high=high, mean=mean, std=std,
            activation_type=activation_type, fan_mode=fan_mode,
            negative_slope=negative_slope)
        if model_mode == ModeType.TRAIN and \
                pretrained_embedding_file is not None and \
                pretrained_embedding_file != "":
            self.load_pretrained_embedding(
                embedding_lookup_table, dict_map, embedding_dim, name,
                pretrained_embedding_file)
        if padding_idx is not None:
            embedding_lookup_table[padding_idx] = 0.0
        self.embedding.weight.data.copy_(embedding_lookup_table)

    def forward(self, vocab_ids, offset=None):
        if self.mode == EmbeddingProcessType.FLAT:
            embedding = self.embedding(vocab_ids)
        else:
            embedding = self.embedding(vocab_ids, offset)
        return self.dropout(embedding)

    def load_pretrained_embedding(
            self, embedding_lookup_table, dict_map, embedding_dim, name,
            pretrained_embedding_file):
        self.logger.warn(
            "Load %s embedding from %s" % (name, pretrained_embedding_file))
        with open(pretrained_embedding_file) as fin:
            num_pretrained = 0
            for line in fin:
                data = line.strip().split(' ')
                # Check embedding info
                if len(data) == 2:
                    assert int(data[1]) == embedding_dim, \
                        "Pretrained embedding dim not matching: %s, %d" % (
                            data[1], embedding_dim)
                    continue
                if data[0] not in dict_map:
                    continue
                embedding = torch.FloatTensor([float(i) for i in data[1:]])
                embedding_lookup_table[dict_map[data[0]]] = embedding
                num_pretrained += 1
        self.logger.warn(
            "Total dict size of %s is %d" % (name, len(dict_map)))
        self.logger.warn("Size of pretrained %s embedding is %d" % (
            name, num_pretrained))
        self.logger.warn(
            "Size of randomly initialize %s embedding is %d" % (
                name, len(dict_map) - num_pretrained))


class RegionEmbeddingType(Type):
    """Standard names for region embedding type
    """
    WC = 'word_context'
    CW = 'context_word'
    
    @classmethod
    def str(cls):
        return ",".join([cls.WC, cls.CW])


class RegionEmbeddingLayer(torch.nn.Module):
    """
    Reference: A New Method of Region Embedding for Text Classification
    """

    def __init__(self, dict_map, embedding_dim, region_size, name, config,
                 padding=None, pretrained_embedding_file=None, dropout=0,
                 init_type=InitType.XAVIER_UNIFORM, low=0, high=1, mean=0,
                 std=1, fan_mode=FAN_MODE.FAN_IN, model_mode=ModeType.TRAIN,
                 region_embedding_type=RegionEmbeddingType.WC):
        super(RegionEmbeddingLayer, self).__init__()
        self.region_embedding_type = region_embedding_type
        self.region_size = region_size
        assert self.region_size % 2 == 1
        self.radius = int(region_size / 2)
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(
            dict_map, embedding_dim, "RegionWord" + name, config=config,
            padding_idx=padding,
            pretrained_embedding_file=pretrained_embedding_file,
            dropout=dropout, init_type=init_type, low=low, high=high, mean=mean,
            std=std, fan_mode=fan_mode, model_mode=model_mode)
        self.context_embedding = Embedding(
            dict_map, embedding_dim * region_size, "RegionContext" + name,
            config=config, padding_idx=padding, dropout=dropout,
            init_type=init_type, low=low, high=high, mean=mean, std=std,
            fan_mode=fan_mode)

    def forward(self, vocab_ids):
        seq_length = vocab_ids.size(1)
        actual_length = vocab_ids.size(1) - self.radius * 2
        trim_vocab_id = vocab_ids[:, self.radius:seq_length - self.radius]
        slice_vocabs = \
            [vocab_ids[:, i:i + self.region_size] for i in
             range(actual_length)]
        slice_vocabs = torch.cat(slice_vocabs, 1)
        slice_vocabs = \
            slice_vocabs.view(-1, actual_length, self.region_size)

        if self.region_embedding_type == RegionEmbeddingType.WC:
            vocab_embedding = self.embedding(slice_vocabs)
            context_embedding = self.context_embedding(trim_vocab_id)
            context_embedding = context_embedding.view(
                -1, actual_length, self.region_size, self.embedding_dim)
            region_embedding = vocab_embedding * context_embedding
            region_embedding, _ = region_embedding.max(2)
        elif self.region_embedding_type == RegionEmbeddingType.CW:
            vocab_embedding = self.embedding(trim_vocab_id).unsqueeze(2)
            context_embedding = self.context_embedding(slice_vocabs)
            size = context_embedding.size()
            context_embedding = context_embedding.view(
                size[0], size[1], size[2], self.region_size, self.embedding_dim)
            mask = torch.ones(
                [self.region_size, self.region_size, self.embedding_dim])

            for i in range(self.region_size):
                mask[i][self.region_size - i - 1] = 0.
            neg_mask = mask * -65500.0
            mask = mask.le(0).float()
            mask = mask.unsqueeze(0).unsqueeze(0)
            context_embedding = context_embedding * mask
            context_embedding = context_embedding + neg_mask
            context_embedding, _ = context_embedding.max(3)
            region_embedding = vocab_embedding * context_embedding
            region_embedding, _ = region_embedding.max(2)
        else:
            raise TypeError(
                "Unsupported region embedding type: %s." %
                self.region_embedding_type)

        return region_embedding

class PositionEmbedding(torch.nn.Module):
    ''' Reference: attention is all you need '''

    def __init__(self, seq_max_len, embedding_dim, padding_idx):
        super(PositionEmbedding, self).__init__()

        self.position_enc = nn.Embedding.from_pretrained(
            self.get_sinusoid_encoding_table(seq_max_len + 1,
                                             embedding_dim,
                                             padding_idx=padding_idx),
            freeze=True)

    def forward(self, src_pos):
        return self.position_enc(src_pos)

    @staticmethod
    def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array(
            [get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return torch.FloatTensor(sinusoid_table)
