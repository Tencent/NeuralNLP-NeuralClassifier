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


from dataset.dataset import DatasetBase
from dataset.dataset import InsertVocabMode
from util import ModeType


class ClassificationDataset(DatasetBase):
    CLASSIFICATION_LABEL_SEPARATOR = "--"
    DOC_LABEL = "doc_label"
    DOC_LABEL_LIST = "doc_label_list"

    DOC_TOKEN = "doc_token"
    DOC_CHAR = "doc_char"
    DOC_CHAR_IN_TOKEN = "doc_char_in_token"
    DOC_TOKEN_NGRAM = "doc_token_ngram"
    DOC_KEYWORD = "doc_keyword"
    DOC_TOPIC = "doc_topic"

    DOC_TOKEN_OFFSET = "doc_token_offset"
    DOC_TOKEN_NGRAM_OFFSET = "doc_token_ngram_offset"
    DOC_KEYWORD_OFFSET = "doc_keyword_offset"
    DOC_TOPIC_OFFSET = "doc_topic_offset"

    DOC_TOKEN_LEN = "doc_token_len"
    DOC_CHAR_LEN = "doc_char_len"
    DOC_CHAR_IN_TOKEN_LEN = "doc_char_in_token_len"
    DOC_TOKEN_NGRAM_LEN = "doc_token_ngram_len"
    DOC_KEYWORD_LEN = "doc_keyword_len"
    DOC_TOPIC_LEN = "doc_topic_len"

    DOC_TOKEN_MASK = "doc_token_mask"
    DOC_CHAR_MASK = "doc_char_mask"
    DOC_CHAR_IN_TOKEN_MASK = "doc_char_in_token_mask"

    DOC_TOKEN_MAX_LEN = "doc_token_max_len"
    DOC_CHAR_MAX_LEN = "doc_char_max_len"
    DOC_CHAR_IN_TOKEN_MAX_LEN = "doc_char_in_token_max_len"

    def __init__(self, config, json_files, generate_dict=False,
                 mode=ModeType.EVAL):
        super(ClassificationDataset, self).__init__(
            config, json_files, generate_dict=generate_dict, mode=mode)

    def _init_dict(self):
        self.dict_names = \
            [self.DOC_LABEL, self.DOC_TOKEN, self.DOC_CHAR,
             self.DOC_TOKEN_NGRAM, self.DOC_KEYWORD, self.DOC_TOPIC]

        self.dict_files = []
        for dict_name in self.dict_names:
            self.dict_files.append(
                self.config.data.dict_dir + "/" + dict_name + ".dict")
        self.label_dict_file = self.dict_files[0]

        # By default keep all labels
        self.min_count = [0,
                          self.config.feature.min_token_count,
                          self.config.feature.min_char_count,
                          self.config.feature.min_token_ngram_count,
                          self.config.feature.min_keyword_count,
                          self.config.feature.min_topic_count]

        # By default keep all labels
        self.max_dict_size = [self.BIG_VALUE,
                              self.config.feature.max_token_dict_size,
                              self.config.feature.max_char_dict_size,
                              self.config.feature.max_token_ngram_dict_size,
                              self.config.feature.max_keyword_dict_size,
                              self.config.feature.max_topic_dict_size]

        self.max_sequence_length = [
            self.config.feature.max_token_len,
            self.config.feature.max_char_len]

        # When generating dict, the following map store vocab count.
        # Then clear dict and load vocab of word index
        self.label_map = dict()
        self.token_map = dict()
        self.char_map = dict()
        self.token_ngram_map = dict()
        self.keyword_map = dict()
        self.topic_map = dict()
        self.dicts = [self.label_map, self.token_map, self.char_map,
                      self.token_ngram_map, self.keyword_map, self.topic_map]

        # Save sorted dict according to the count
        self.label_count_list = []
        self.token_count_list = []
        self.char_count_list = []
        self.token_ngram_count_list = []
        self.keyword_count_list = []
        self.topic_count_list = []
        self.count_list = [self.label_count_list, self.token_count_list,
                           self.char_count_list, self.token_ngram_count_list,
                           self.keyword_count_list, self.topic_count_list]

        self.id_to_label_map = dict()
        self.id_to_token_map = dict()
        self.id_to_char_map = dict()
        self.id_to_token_gram_map = dict()
        self.id_to_keyword_map = dict()
        self.id_to_topic_map = dict()
        self.id_to_vocab_dict_list = [
            self.id_to_label_map, self.id_to_token_map, self.id_to_char_map,
            self.id_to_token_gram_map, self.id_to_keyword_map,
            self.id_to_topic_map]

        self.pretrained_dict_names = [self.DOC_TOKEN, self.DOC_KEYWORD]
        self.pretrained_dict_files = \
            [self.config.feature.token_pretrained_file,
             self.config.feature.keyword_pretrained_file]
        self.pretrained_min_count = \
            [self.config.feature.min_token_count,
             self.config.feature.min_keyword_count]

    def _insert_vocab(self, json_obj, mode=InsertVocabMode.ALL):
        """Insert vocab to dict
        """
        if mode == InsertVocabMode.ALL or mode == InsertVocabMode.LABEL:
            doc_labels = json_obj[self.DOC_LABEL]
            self._insert_sequence_vocab(doc_labels, self.label_map)
        if mode == InsertVocabMode.ALL or mode == InsertVocabMode.OTHER:
            doc_tokens = \
                json_obj[self.DOC_TOKEN][0:self.config.feature.max_token_len]
            doc_keywords = json_obj[self.DOC_KEYWORD]
            doc_topics = json_obj[self.DOC_TOPIC]

            self._insert_sequence_tokens(
                doc_tokens, self.token_map, self.token_ngram_map, self.char_map,
                self.config.feature.token_ngram)
            self._insert_sequence_vocab(doc_keywords, self.keyword_map)
            self._insert_sequence_vocab(doc_topics, self.topic_map)

    def _get_vocab_id_list(self, json_obj):
        """Use dict to convert all vocabs to ids
        """
        doc_labels = json_obj[self.DOC_LABEL]
        doc_tokens = \
            json_obj[self.DOC_TOKEN][0:self.config.feature.max_token_len]
        doc_keywords = json_obj[self.DOC_KEYWORD]
        doc_topics = json_obj[self.DOC_TOPIC]

        token_ids, char_ids, char_in_token_ids, token_ngram_ids = \
            self._token_to_id(doc_tokens, self.token_map, self.char_map,
                              self.config.feature.token_ngram,
                              self.token_ngram_map,
                              self.config.feature.max_char_len,
                              self.config.feature.max_char_len_per_token)
        return {self.DOC_LABEL: self._label_to_id(doc_labels, self.label_map) if self.model_mode != ModeType.PREDICT else [0],
                self.DOC_TOKEN: token_ids, self.DOC_CHAR: char_ids,
                self.DOC_CHAR_IN_TOKEN: char_in_token_ids,
                self.DOC_TOKEN_NGRAM: token_ngram_ids,
                self.DOC_KEYWORD:
                    self._vocab_to_id(doc_keywords, self.keyword_map),
                self.DOC_TOPIC: self._vocab_to_id(doc_topics, self.topic_map)}
