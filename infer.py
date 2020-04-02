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

    def threshold(self,dev_df:pd.DataFrame,recall_require=0.99,precision_require=0.99, recall_col="负面",precision_col='正面',text_col="信息标题",label_col = '情感'):
        print(dev_df[text_col])    
        emotions = self.infer(dev_df[text_col].tolist())
        print(emotions)
        time.sleep(20)
        emotions_list = [[x['中性'],x['正面'],x['负面']] for x in emotions]
        dev_df["中性"] = None
        dev_df['正面'] = None
        dev_df['负面'] = None
        dev_df[['中性','正面','负面']] = emotions_list



        pos_thresh = 0.73
        neg_thresh = 0.1
        best_pos_thresh = -1
        best_neg_thresh = -1
        best_precision = -1
        best_recall = -1
        collections = []
        steps = 100
        step_size = 0.01/steps

        for i in range(steps):
            pos_thresh+=step_size
            neg_thresh = 0.1
            for j in range(steps+1):
                print(pos_thresh)
                print(neg_thresh)
                neg_thresh -=step_size
                pred = self.infer_class(emotions,pos_thresh,neg_thresh)
                report = self.eval_pred(pred,dev_df[label_col])
                if report[precision_col]['precision']>best_precision and report[precision_col]['precision'] > precision_require and report[recall_col]["recall"] >best_recall and report[recall_col]['recall'] > recall_require:
                    best_precision = report[precision_col]['precision']
                    best_pos_thresh = pos_thresh
                    best_recall = report[recall_col]['recall']
                    best_neg_thresh = neg_thresh
                    collections.append({"best_pos_thresh":pos_thresh,"best_neg_thresh":neg_thresh,"pos_precision":best_precision,"neg_recall":best_recall})
            if report[precision_col]['precision']<0.5:
                break
        return collections


    def eval_pred(self,pred,true):
        report = classification_report(y_true = true, y_pred = pred,output_dict = True)
        print(report)
        print(
        return report
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

    collections= inferer.threshold(dev_df)
    print(collections)
