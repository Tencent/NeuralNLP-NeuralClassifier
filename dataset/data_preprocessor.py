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
import pandas as pd
import random
from tokenizer import get_tokenizer
from tqdm import tqdm


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


def preprocess(sample:dict, tokenizer:None):
    sample['doc_label'] = sample['doc_label']
    if not tokenizer:
        sample['doc_token'] = text_data.split(" ")
    else:
        sample['doc_token'] = tokenizer.tokenize(sample['doc_text'])['doc_token']
    sample['doc_keyword'] = []
    sample['doc_topic'] = []
    return sample



def read_csv(excel_file:str):
    name = excel_file.split('.')[0]
    table = pd.read_excel(excel_file)
    file_train = open(f"{name}_train.json", 'w')
    file_test  = open(f"{name}_test.json", 'w')
    file_dev   = open(f"{name}_dev.json", 'w')
    tokenizer = get_tokenizer()
    
    dataset = []
    for i in tqdm(range(len(table))):
        items = table.iloc[i].to_dict()
        if items['情感'] in ['有害', '关注']:
            continue

        items = {
                'doc_text': items['信息标题'],
                'doc_label': [items['情感'].strip()]
                }
        items = preprocess(items, tokenizer)
        dataset.append(items)
        json_str = json.dumps(items, ensure_ascii=False)
        if random.random() < 0.1:
            file_test.write(json_str+'\n')
        elif random.random() < 0.2:
            file_dev.write(json_str+'\n')
        else:
            file_train.write(json_str+'\n')


if __name__ == '__main__':
    csv_file = sys.argv[1]
    read_csv(csv_file)
