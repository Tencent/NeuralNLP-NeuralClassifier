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

import codecs
import math
import numpy as np
import os
import sys
import json

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset.classification_dataset import ClassificationDataset
from dataset.collator import ClassificationCollator
from dataset.collator import ClassificationType
from dataset.collator import FastTextCollator
from model.classification.drnn import DRNN
from model.classification.fasttext import FastText
from model.classification.textcnn import TextCNN
from model.classification.textvdcnn import TextVDCNN
from model.classification.textrnn import TextRNN
from model.classification.textrcnn import TextRCNN
from model.classification.transformer import Transformer
from model.classification.dpcnn import DPCNN
from model.classification.attentive_convolution import AttentiveConvNet
from model.classification.region_embedding import RegionEmbedding
from model.model_util import get_optimizer, get_hierar_relations

ClassificationDataset, ClassificationCollator, FastTextCollator,FastText, TextCNN, TextRNN, TextRCNN, DRNN, TextVDCNN, Transformer, DPCNN, AttentiveConvNet, RegionEmbedding

class Predictor(object):
    def __init__(self, config):
        self.config = config
        self.model_name = config.model_name
        self.use_cuda = config.device.startswith("cuda")
        self.dataset_name = "ClassificationDataset"
        self.collate_name = "FastTextCollator" if self.model_name == "FastText" \
                else "ClassificationCollator"
        self.dataset = globals()[self.dataset_name](config, [], mode="infer")
        self.collate_fn = globals()[self.collate_name](config, len(self.dataset.label_map))
        self.model = Predictor._get_classification_model(self.model_name, self.dataset, config)
        Predictor._load_checkpoint(config.eval.model_dir, self.model, self.use_cuda)
        self.model.eval()
        
    @staticmethod
    def _get_classification_model(model_name, dataset, conf):
        model = globals()[model_name](dataset, conf)
        model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
        return model

    @staticmethod
    def _load_checkpoint(file_name, model, use_cuda):
        if use_cuda:
            checkpoint = torch.load(file_name)
        else:
            checkpoint = torch.load(file_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
    
    def predict(self, texts):
        """
        input texts should be json objects
        """
        with torch.no_grad():
            batch_texts = [self.dataset._get_vocab_id_list(json.loads(text)) for text in texts]
            batch_texts = self.collate_fn(batch_texts)
            logits = self.model(batch_texts)
            if self.config.task_info.label_type != ClassificationType.MULTI_LABEL:
                probs = torch.softmax(logits, dim=1)
            else:
                probs = torch.sigmoid(logits)
            probs = probs.cpu().tolist()
            return np.array(probs)

if __name__ == "__main__":
    config = Config(config_file=sys.argv[1])
    predictor = Predictor(config)
    batch_size = config.eval.batch_size
    input_texts = []
    predict_probs = []
    is_multi = config.task_info.label_type == ClassificationType.MULTI_LABEL
    for line in codecs.open(sys.argv[2], "r", predictor.dataset.CHARSET):
        input_texts.append(line.strip("\n"))
    epoches = math.ceil(len(input_texts)/batch_size)
    for i in range(epoches):
        batch_texts = input_texts[i*batch_size:(i+1)*batch_size]
        predict_prob = predictor.predict(batch_texts)
        for j in predict_prob:
            predict_probs.append(j)
    with codecs.open("predict.txt", "w", predictor.dataset.CHARSET) as of:
        for predict_prob in predict_probs:
            if not is_multi:
                predict_label_ids = [predict_prob.argmax()]
            else:
                predict_label_ids = []
                predict_label_idx = np.argsort(-predict_prob)
                for j in range(0, config.eval.top_k):
                    if predict_prob[predict_label_idx[j]] > config.eval.threshold:
                        predict_label_ids.append(predict_label_idx[j])
            predict_label_name = [predictor.dataset.id_to_label_map[predict_label_id] \
                    for predict_label_id in predict_label_ids]
            of.write(";".join(predict_label_name) + "\n") 
