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
import torch.nn as nn

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.transformer_encoder import EncoderLayer, StarEncoderLayer
from model.embedding import PositionEmbedding


class Transformer(Classifier):
    def __init__(self, dataset, config):
        super(Transformer, self).__init__(dataset, config)

        self.pad = dataset.token_map[dataset.VOCAB_PADDING]

        if config.feature.feature_names[0] == "token":
            seq_max_len = config.feature.max_token_len
        else:
            seq_max_len = config.feature.max_char_len
        self.position_enc = PositionEmbedding(seq_max_len,
                                              config.embedding.dimension,
                                              self.pad)

        if config.Transformer.use_star:
            self.layer_stack = nn.ModuleList([
                StarEncoderLayer(config.embedding.dimension,
                                 config.Transformer.n_head,
                                 config.Transformer.d_k,
                                 config.Transformer.d_v,
                                 dropout=config.Transformer.dropout)
                for _ in range(config.Transformer.n_layers)])
        else:
            self.layer_stack = nn.ModuleList([
                EncoderLayer(config.embedding.dimension,
                             config.Transformer.d_inner,
                             config.Transformer.n_head,
                             config.Transformer.d_k,
                             config.Transformer.d_v,
                             dropout=config.Transformer.dropout)
                for _ in range(config.Transformer.n_layers)])

        hidden_size = config.embedding.dimension
        self.linear = torch.nn.Linear(hidden_size, len(dataset.label_map))
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

    def get_parameter_optimizer_dict(self):
        params = list()
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.char_embedding.parameters()})
        for i in range(0, len(self.layer_stack)):
            params.append({'params': self.layer_stack[i].parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def update_lr(self, optimizer, epoch):
        if epoch > self.config.train.num_epochs_static_embedding:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = self.config.optimizer.learning_rate
        else:
            for param_group in optimizer.param_groups[:2]:
                param_group["lr"] = 0

    def forward(self, batch):
        def _get_non_pad_mask(seq, pad):
            assert seq.dim() == 2
            return seq.ne(pad).type(torch.float).unsqueeze(-1)

        def _get_attn_key_pad_mask(seq_k, seq_q, pad):
            ''' For masking out the padding part of key sequence. '''

            # Expand to fit the shape of key query attention matrix.
            len_q = seq_q.size(1)
            padding_mask = seq_k.eq(pad)
            padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

            return padding_mask

        if self.config.feature.feature_names[0] == "token":
            src_seq = batch[cDataset.DOC_TOKEN].to(self.config.device)
            embedding = self.token_embedding(src_seq)
        else:
            src_seq = batch[cDataset.DOC_CHAR].to(self.config.device)
            embedding = self.char_embedding(src_seq)

        # Prepare masks
        slf_attn_mask = _get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad=self.pad)
        non_pad_mask = _get_non_pad_mask(src_seq, self.pad)

        batch_lens = (src_seq != self.pad).sum(dim=-1)
        src_pos = torch.zeros_like(src_seq, dtype=torch.long)
        for row, length in enumerate(batch_lens):
            src_pos[row][:length] = torch.arange(1, length + 1)

        enc_output = embedding + self.position_enc(src_pos)

        if self.config.Transformer.use_star:
            s = torch.mean(embedding, 1)  # virtual relay node
            h = enc_output
            for enc_layer in self.layer_stack:
                h, s = enc_layer(h, embedding, s,
                                 non_pad_mask=non_pad_mask,
                                 slf_attn_mask=None)
            h_max, _ = torch.max(h, 1)
            enc_output = h_max + s
        else:
            for enc_layer in self.layer_stack:
                enc_output, _ = enc_layer(enc_output,
                                          non_pad_mask=non_pad_mask,
                                          slf_attn_mask=slf_attn_mask)
            enc_output = torch.mean(enc_output, 1)

        return self.dropout(self.linear(enc_output))
