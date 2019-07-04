#!/usr/bin/env python
#coding:utf-8
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

"""Collator for NeuralClassifier"""

import torch

from dataset.classification_dataset import ClassificationDataset as cDataset
from util import Type


class Collator(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        raise NotImplementedError


class ClassificationType(Type):
    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"

    @classmethod
    def str(cls):
        return ",".join([cls.SINGLE_LABEL, cls.MULTI_LABEL])


class ClassificationCollator(Collator):
    def __init__(self, conf, label_size):
        super(ClassificationCollator, self).__init__(conf.device)
        self.classification_type = conf.task_info.label_type
        min_seq = 1
        if conf.model_name == "TextCNN":
            min_seq = conf.TextCNN.top_k_max_pooling
        elif conf.model_name == "DPCNN":
            min_seq = conf.DPCNN.kernel_size * 2 ** conf.DPCNN.blocks
        elif conf.model_name == "RegionEmbedding":
            min_seq = conf.feature.max_token_len
        self.min_token_max_len = min_seq
        self.min_char_max_len = min_seq
        self.label_size = label_size

    def _get_multi_hot_label(self, doc_labels):
        """For multi-label classification
        Generate multi-hot for input labels
        e.g. input: [[0,1], [2]]
             output: [[1,1,0], [0,0,1]]
        """
        batch_size = len(doc_labels)
        max_label_num = max([len(x) for x in doc_labels])
        doc_labels_extend = \
            [[doc_labels[i][0] for x in range(max_label_num)] for i in range(batch_size)]
        for i in range(0, batch_size):
            doc_labels_extend[i][0 : len(doc_labels[i])] = doc_labels[i]
        y = torch.Tensor(doc_labels_extend).long()
        y_onehot = torch.zeros(batch_size, self.label_size).scatter_(1, y, 1)
        return y_onehot

    def _append_label(self, doc_labels, sample):
        if self.classification_type == ClassificationType.SINGLE_LABEL:
            assert len(sample[cDataset.DOC_LABEL]) == 1
            doc_labels.extend(sample[cDataset.DOC_LABEL])
        elif self.classification_type == ClassificationType.MULTI_LABEL:
            doc_labels.append(sample[cDataset.DOC_LABEL])
        else:
            raise TypeError(
                "Unsupported classification type: %s. Supported "
                "classification type is: %s" %
                (self.classification_type, ClassificationType.str()))

    def __call__(self, batch):
        def _append_vocab(ori_vocabs, vocabs, max_len):
            padding = [cDataset.VOCAB_PADDING] * (max_len - len(ori_vocabs))
            vocabs.append(ori_vocabs + padding)

        doc_labels = []

        doc_token = []
        doc_char = []
        doc_char_in_token = []

        doc_token_len = []
        doc_char_len = []
        doc_char_in_token_len = []

        doc_token_max_len = self.min_token_max_len
        doc_char_max_len = self.min_char_max_len
        doc_char_in_token_max_len = 0

        for _, value in enumerate(batch):
            doc_token_max_len = max(doc_token_max_len,
                                    len(value[cDataset.DOC_TOKEN]))
            doc_char_max_len = max(doc_char_max_len,
                                   len(value[cDataset.DOC_CHAR]))
            for char_in_token in value[cDataset.DOC_CHAR_IN_TOKEN]:
                doc_char_in_token_max_len = max(doc_char_in_token_max_len,
                                                len(char_in_token))

        for _, value in enumerate(batch):
            self._append_label(doc_labels, value)
            _append_vocab(value[cDataset.DOC_TOKEN], doc_token,
                          doc_token_max_len)
            doc_token_len.append(len(value[cDataset.DOC_TOKEN]))
            _append_vocab(value[cDataset.DOC_CHAR], doc_char, doc_char_max_len)
            doc_char_len.append(len(value[cDataset.DOC_CHAR]))

            doc_char_in_token_len_tmp = []
            for char_in_token in value[cDataset.DOC_CHAR_IN_TOKEN]:
                _append_vocab(char_in_token, doc_char_in_token,
                              doc_char_in_token_max_len)
                doc_char_in_token_len_tmp.append(len(char_in_token))

            padding = [cDataset.VOCAB_PADDING] * doc_char_in_token_max_len
            for _ in range(
                    len(value[cDataset.DOC_CHAR_IN_TOKEN]), doc_token_max_len):
                doc_char_in_token.append(padding)
                doc_char_in_token_len_tmp.append(0)
            doc_char_in_token_len.append(doc_char_in_token_len_tmp)

        if self.classification_type == ClassificationType.SINGLE_LABEL:
            tensor_doc_labels = torch.tensor(doc_labels)
            doc_label_list = [[x] for x in doc_labels]
        elif self.classification_type == ClassificationType.MULTI_LABEL:
            tensor_doc_labels = self._get_multi_hot_label(doc_labels)
            doc_label_list = doc_labels

        batch_map = {
            cDataset.DOC_LABEL: tensor_doc_labels,
            cDataset.DOC_LABEL_LIST: doc_label_list,

            cDataset.DOC_TOKEN: torch.tensor(doc_token),
            cDataset.DOC_CHAR: torch.tensor(doc_char),
            cDataset.DOC_CHAR_IN_TOKEN: torch.tensor(doc_char_in_token),

            cDataset.DOC_TOKEN_MASK: torch.tensor(doc_token).gt(0).float(),
            cDataset.DOC_CHAR_MASK: torch.tensor(doc_char).gt(0).float(),
            cDataset.DOC_CHAR_IN_TOKEN_MASK:
                torch.tensor(doc_char_in_token).gt(0).float(),

            cDataset.DOC_TOKEN_LEN: torch.tensor(
                doc_token_len, dtype=torch.float32),
            cDataset.DOC_CHAR_LEN: torch.tensor(
                doc_char_len, dtype=torch.float32),
            cDataset.DOC_CHAR_IN_TOKEN_LEN: torch.tensor(
                doc_char_in_token_len, dtype=torch.float32),

            cDataset.DOC_TOKEN_MAX_LEN:
                torch.tensor([doc_token_max_len], dtype=torch.float32),
            cDataset.DOC_CHAR_MAX_LEN:
                torch.tensor([doc_char_max_len], dtype=torch.float32),
            cDataset.DOC_CHAR_IN_TOKEN_MAX_LEN:
                torch.tensor([doc_char_in_token_max_len], dtype=torch.float32)
        }
        return batch_map


class FastTextCollator(ClassificationCollator):
    """FastText Collator
    Extra support features: token, token-ngrams, keywords, topics.
    """
    def __call__(self, batch):
        def _append_vocab(sample, vocabs, offsets, lens, name):
            filtered_vocab = [x for x in sample[name] if
                              x is not cDataset.VOCAB_UNKNOWN]
            vocabs.extend(filtered_vocab)
            offsets.append(offsets[-1] + len(filtered_vocab))
            lens.append(len(filtered_vocab))

        doc_labels = []

        doc_tokens = []
        doc_token_ngrams = []
        doc_keywords = []
        doc_topics = []

        doc_tokens_offset = [0]
        doc_token_ngrams_offset = [0]
        doc_keywords_offset = [0]
        doc_topics_offset = [0]

        doc_tokens_len = []
        doc_token_ngrams_len = []
        doc_keywords_len = []
        doc_topics_len = []
        for _, value in enumerate(batch):
            self._append_label(doc_labels, value)
            _append_vocab(value, doc_tokens, doc_tokens_offset,
                          doc_tokens_len,
                          cDataset.DOC_TOKEN)
            _append_vocab(value, doc_token_ngrams, doc_token_ngrams_offset,
                          doc_token_ngrams_len,
                          cDataset.DOC_TOKEN_NGRAM)
            _append_vocab(value, doc_keywords, doc_keywords_offset,
                          doc_keywords_len, cDataset.DOC_KEYWORD)
            _append_vocab(value, doc_topics, doc_topics_offset,
                          doc_topics_len, cDataset.DOC_TOPIC)
        doc_tokens_offset.pop()
        doc_token_ngrams_offset.pop()
        doc_keywords_offset.pop()
        doc_topics_offset.pop()

        if self.classification_type == ClassificationType.SINGLE_LABEL:
            tensor_doc_labels = torch.tensor(doc_labels)
            doc_label_list = [[x] for x in doc_labels]
        elif self.classification_type == ClassificationType.MULTI_LABEL:
            tensor_doc_labels = self._get_multi_hot_label(doc_labels)
            doc_label_list = doc_labels

        batch_map = {
            cDataset.DOC_LABEL: tensor_doc_labels,
            cDataset.DOC_LABEL_LIST: doc_label_list,

            cDataset.DOC_TOKEN: torch.tensor(doc_tokens),
            cDataset.DOC_TOKEN_NGRAM: torch.tensor(doc_token_ngrams),
            cDataset.DOC_KEYWORD: torch.tensor(doc_keywords),
            cDataset.DOC_TOPIC: torch.tensor(doc_topics),

            cDataset.DOC_TOKEN_OFFSET: torch.tensor(doc_tokens_offset),
            cDataset.DOC_TOKEN_NGRAM_OFFSET:
                torch.tensor(doc_token_ngrams_offset),
            cDataset.DOC_KEYWORD_OFFSET: torch.tensor(doc_keywords_offset),
            cDataset.DOC_TOPIC_OFFSET: torch.tensor(doc_topics_offset),

            cDataset.DOC_TOKEN_LEN:
                torch.tensor(doc_tokens_len, dtype=torch.float32),
            cDataset.DOC_TOKEN_NGRAM_LEN:
                torch.tensor(doc_token_ngrams_len, dtype=torch.float32),
            cDataset.DOC_KEYWORD_LEN:
                torch.tensor(doc_keywords_len, dtype=torch.float32),
            cDataset.DOC_TOPIC_LEN:
                torch.tensor(doc_topics_len, dtype=torch.float32)}
        return batch_map
