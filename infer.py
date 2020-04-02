from predict import Predictor

from dataset.tokenizer import get_tokenizer
from dataset.classification_dataset import ClassificationDataset
import json
from config import Config
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import time
class Inferer():
    def __init__(self,conf):
        self.predictor = Predictor(conf)
        self.tokenizer = get_tokenizer("jieba")
        self.empty_dataset = globals()["ClassificationDataset"](conf,[],mode = 'train')
        self.label_map = self.empty_dataset.label_map

    def infer(self,texts):
        objs = [json.dumps(self.tokenizer.tokenize(x),ensure_ascii = False) for x in texts]
        objs = []
        for text in texts:
            datapoint = self.tokenizer.tokenize(text)
            datapoint['doc_label'] = []
            datapoint['doc_topic'] = []
            datapoint['doc_keyword'] = []

            objs.append(json.dumps(datapoint,ensure_ascii=False))
       
        logits = self.predictor.predict(objs)

    
        scores = (lambda x:1/(1+np.exp(-x)))(logits)
        out = [{"正面":x[self.label_map["正面"]],
                "负面":x[self.label_map["负面"]],
                "中性":x[self.label_map["中性"]]} for x in scores]
        return out
    
    def infer_class(self,logits,pos_thresh,neg_thresh):
        pred = []
        for logit in logits:
            if logit['正面']>pos_thresh or logit['负面']<neg_thresh:
                pred.append("正面")
            else:
                pred.append("负面")
        return pred

if __name__=="__main__":

    '''
        infer: python infer.py conf/train.json
    '''
    f = open("./data/sentiment_classification_dev.json",'r')
    dev_df = []
    for line in f.readlines():
        datapoint = json.loads(line)
        dev_df.append([''.join(datapoint['doc_token']),datapoint['doc_label'][0]])
    dev_df = pd.DataFrame(dev_df)
    dev_df.columns = ['信息标题','情感']
    
    config = Config(config_file=sys.argv[1])
    inferer = Inferer(config)
   
