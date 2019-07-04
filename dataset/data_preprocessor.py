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


# Same as "A New Method of Region Embedding for Text Classification"
# https://github.com/text-representation/local-context-unit/blob/master/bin/prepare.py

import csv
import json
import re
import sys


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " \. ", string)
    string = re.sub(r"\"", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def convert_multi_slots_to_single_slots(slots):
    """
    covert the data which text_data are saved as multi-slots, e.g()
    """
    if len(slots) == 1:
        return slots[0]
    else:
        return ' '.join(slots)


def preprocess(csv_file, json_file):
    with open(json_file, "w") as fout:
        with open(csv_file, 'rb') as fin:
            lines = csv.reader(fin)
            for items in lines:
                text_data = convert_multi_slots_to_single_slots(items[1:])
                text_data = clean_str(text_data)
                sample = dict()
                sample['doc_label'] = [items[0]]
                sample['doc_token'] = text_data.split(" ")
                sample['doc_keyword'] = []
                sample['doc_topic'] = []
                json_str = json.dumps(sample, ensure_ascii=False)
                fout.write(json_str)


if __name__ == '__main__':
    train_csv = sys.argv[1]
    train_json = sys.argv[2]
    test_csv = sys.argv[3]
    test_json = sys.argv[4]
    preprocess(train_csv, train_json)
    preprocess(test_csv, test_json)
