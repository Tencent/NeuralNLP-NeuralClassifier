#!/usr/bin/env python
#coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 Tencent. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

# Provide function that calculate the precision, recall, F1-score
#   and output confusion_matrix.


import json
import os

import numpy as np

from dataset.classification_dataset import ClassificationDataset as cDataset


class ClassificationEvaluator(object):
    MACRO_AVERAGE = "macro_average"
    MICRO_AVERAGE = "micro_average"
    """Not thread safe, will keep the latest eval result
    """

    def __init__(self, eval_dir):
        self.confusion_matrix_list = None
        self.precision_list = None
        self.recall_list = None
        self.fscore_list = None
        self.right_list = None
        self.predict_list = None
        self.standard_list = None

        self.eval_dir = eval_dir
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

    @staticmethod
    def _calculate_prf(right_count, predict_count, standard_count):
        """Calculate precision, recall, fscore
        Args:
            standard_count: Standard count
            predict_count: Predict count
            right_count: Right count
        Returns:
            precision, recall, f_score
        """
        precision, recall, f_score = 0, 0, 0
        if predict_count > 0:
            precision = right_count / predict_count
        if standard_count > 0:
            recall = right_count / standard_count
        if precision + recall > 0:
            f_score = precision * recall * 2 / (precision + recall)

        return precision, recall, f_score

    @staticmethod
    def _judge_label_in(label_name, label_to_id_maps):
        cnt = 0
        for label in label_name:
            for i in range(0, len(label_to_id_maps)):
                if label in label_to_id_maps[i]:
                    cnt += 1
                    break
        return cnt == len(label_name)
    def calculate_level_performance(
            self, id_to_label_map, right_count_category, predict_count_category,
            standard_count_category, other_text='其他',
            exclude_method="contain"):
        """Calculate the level performance.
        Args:
            id_to_label_map: Label id to label name.
            other_text: Text to judge the other label.
            right_count_category: Right count.
            predict_count_category: Predict count.
            standard_count_category: Standard count.
            exclude_method: The method to judge the other label. Can be
                            contain(label_name contains other_text) or
                            start(label_name start with other_text).
        Returns:
            precision_dict, recall_dict, fscore_dict.
        """
        other_label = dict()
        for _, label_name in id_to_label_map.items():
            if exclude_method == "contain":
                if other_text in label_name:
                    other_label[label_name] = 1
            elif exclude_method == "start":
                if label_name.startswith(other_text):
                    other_label[label_name] = 1
            else:
                raise TypeError(
                    "Cannot find exclude_method: " +
                    exclude_method)

        precision_dict = dict()
        recall_dict = dict()
        fscore_dict = dict()
        precision_dict[self.MACRO_AVERAGE] = 0
        recall_dict[self.MACRO_AVERAGE] = 0
        fscore_dict[self.MACRO_AVERAGE] = 0
        right_total = 0
        predict_total = 0
        standard_total = 0

        for _, label_name in id_to_label_map.items():
            if label_name in other_label:
                continue
            precision_dict[label_name], recall_dict[label_name], \
                fscore_dict[label_name] = self._calculate_prf(
                    right_count_category[label_name],
                    predict_count_category[label_name],
                    standard_count_category[label_name])
            right_total += right_count_category[label_name]
            predict_total += predict_count_category[label_name]
            standard_total += standard_count_category[label_name]
            precision_dict[self.MACRO_AVERAGE] += precision_dict[label_name]
            recall_dict[self.MACRO_AVERAGE] += recall_dict[label_name]
            fscore_dict[self.MACRO_AVERAGE] += fscore_dict[label_name]
        num_label_eval = len(id_to_label_map) - len(other_label)

        precision_dict[self.MACRO_AVERAGE] = \
            precision_dict[self.MACRO_AVERAGE] / num_label_eval
        recall_dict[self.MACRO_AVERAGE] = \
            recall_dict[self.MACRO_AVERAGE] / num_label_eval
        fscore_dict[self.MACRO_AVERAGE] = 0 \
            if (recall_dict[self.MACRO_AVERAGE] +
                precision_dict[self.MACRO_AVERAGE]) == 0 else \
            2 * precision_dict[self.MACRO_AVERAGE] * \
            recall_dict[self.MACRO_AVERAGE] / \
            (recall_dict[self.MACRO_AVERAGE]
             + precision_dict[self.MACRO_AVERAGE])

        right_count_category[self.MICRO_AVERAGE] = right_total
        predict_count_category[self.MICRO_AVERAGE] = predict_total
        standard_count_category[self.MICRO_AVERAGE] = standard_total

        (precision_dict[self.MICRO_AVERAGE], recall_dict[self.MICRO_AVERAGE],
         fscore_dict[self.MICRO_AVERAGE]) = \
            self._calculate_prf(right_total, predict_total, standard_total)
        return precision_dict, recall_dict, fscore_dict

    def evaluate(self, predicts, standard_label_names=None,
                 standard_label_ids=None, label_map=None, threshold=0, top_k=3,
                 is_prob=True, is_flat=False, is_multi=False, other_text='其他'):
        """Eval the predict result.
        Args:
            predicts: Predict probability or
                      predict text label(is_prob is false)
                      fmt:
                      if is_multi: [[p1,p2],[p2],[p3], ...]
                      else: [[p1], [p2], [p3], ...]
            standard_label_names: Standard label names. If standard_label_names
                is None, standard_label_ids should be given.
            standard_label_ids: Standard label ids. If standard_label_ids
                is None, standard_label_names should be given.
            label_map: Label dict. If is_prob is false and label_map is None,
                       label_map will be generated using labels.
            threshold: Threshold to filter probs.
            top_k: if is_multi true, top_k is used for truncating the predicts.
            is_prob: The predict is prob list or label id.
            is_flat: If true, only calculate flat result.
                     Else, calculate hierarchical result.
            is_multi: multi-label evaluation.
            other_text: Label name contains other_text will not be calculate.
        Returns:
            confusion_matrix_list contain all result,
            filtered_confusion_matrix_list contains result that max predict prob
                is greater than threshold and will be used to calculate prf,
            precision_list, recall_list, fscore_list,
            right_count_list, predict_count_list, standard_count_list
        """

        def _init_confusion_matrix(label_map):
            """Init confusion matrix.
            Args:
                label_map: Label map.
            Returns:
                confusion_matrix.
            """
            confusion_matrix = dict()
            for label_name in label_map.keys():
                confusion_matrix[label_name] = dict()
                for label_name_other in label_map.keys():
                    confusion_matrix[label_name][label_name_other] = 0
            return confusion_matrix

        def _init_count_dict(label_map):
            """Init count dict.
            Args:
                label_map: Label map.
            Returns:
                count_dict.
            """
            count_dict = dict()
            for label_name in label_map.keys():
                count_dict[label_name] = 0
            return count_dict

        assert (standard_label_names is not None or
                standard_label_ids is not None)
        sep = cDataset.CLASSIFICATION_LABEL_SEPARATOR
        depth = 0
        if not is_prob and label_map is None:
            assert standard_label_names is not None
            label_map = dict()
            # Use standard_label_names to generate label_map
            for label_list in standard_label_names:
                for label in label_list:
                    if label not in label_map:
                        label_map[label] = len(label_map)
        if not is_flat:
            for label in label_map.keys():
                hierarchical_labels = label.split(sep)
                depth = max(len(hierarchical_labels), depth)
        label_to_id_maps = []
        id_to_label_maps = []
        for i in range(depth + 1):
            label_to_id_maps.append(dict())
            id_to_label_maps.append(dict())
        for label_name, label_id in label_map.items():
            label_to_id_maps[0][label_name] = label_id
            id_to_label_maps[0][label_id] = label_name
            if not is_flat:
                hierarchical_labels = label_name.split(sep)
                for i in range(1, len(hierarchical_labels) + 1):
                    label = sep.join(hierarchical_labels[:i])
                    if label not in label_to_id_maps[i]:
                        index = len(label_to_id_maps[i])
                        label_to_id_maps[i][label] = index
                        id_to_label_maps[i][index] = label

        confusion_matrix_list = []
        right_category_count_list = []
        predict_category_count_list = []
        standard_category_count_list = []
        for i in range(depth + 1):
            confusion_matrix_list.append(
                _init_confusion_matrix(label_to_id_maps[i]))
            right_category_count_list.append(
                _init_count_dict(label_to_id_maps[i]))
            predict_category_count_list.append(
                _init_count_dict(label_to_id_maps[i]))
            standard_category_count_list.append(
                _init_count_dict(label_to_id_maps[i]))

        line_count = 0
        debug_file = open("probs.txt", "w", encoding=cDataset.CHARSET)
        for predict in predicts:
            if is_prob:
                prob_np = np.array(predict, dtype=np.float32)
                if not is_multi:
                    predict_label_ids = [prob_np.argmax()]
                else:
                    predict_label_ids = []
                    predict_label_idx = np.argsort(-prob_np)
                    for j in range(0, top_k):
                        if prob_np[predict_label_idx[j]] > threshold:
                            predict_label_ids.append(predict_label_idx[j])

                predict_label_name = [id_to_label_maps[0][predict_label_id] \
                    for predict_label_id in predict_label_ids]
                debug_file.write(json.dumps(prob_np.tolist()))
                debug_file.write("\n")
            else:
                predict_label_name = predict

            if standard_label_names is not None:
                standard_label_name = standard_label_names[line_count]
            else:
                standard_label_name = [id_to_label_maps[0][standard_label_ids[line_count][i]] \
                    for i in range(len(standard_label_ids[line_count]))]
            if (not self. _judge_label_in(predict_label_name, label_to_id_maps)) or \
                (not self._judge_label_in(standard_label_name, label_to_id_maps)):
                line_count += 1
                continue
            for std_name in standard_label_name:
                for pred_name in predict_label_name:
                    confusion_matrix_list[0][std_name][pred_name] += 1
            for pred_name in predict_label_name:
                predict_category_count_list[0][pred_name] += 1
            for std_name in standard_label_name:
                standard_category_count_list[0][std_name] += 1
                for pred_name in predict_label_name:
                    if std_name == pred_name:
                        right_category_count_list[0][pred_name] += 1

            if not is_flat:
                standard_hierarchical_labels = \
                    [std_name.split(sep) for std_name in standard_label_name]
                predict_hierarchical_labels = \
                    [pred_name.split(sep) for pred_name in predict_label_name]

                standard_label_map = {}
                predict_label_map = {}
                for std_label in standard_hierarchical_labels:
                    for i in range(0, len(std_label)):
                        if i + 1 not in standard_label_map:
                            standard_label_map[i + 1] = set()
                        standard_label_map[i + 1].add(sep.join(std_label[:i+1]))
                for pred_label in predict_hierarchical_labels:
                    for i in range(0, len(pred_label)):
                        if i + 1 not in predict_label_map:
                            predict_label_map[i + 1] = set()
                        predict_label_map[i + 1].add(sep.join(pred_label[:i+1]))
                for level, std_label_set in standard_label_map.items():
                    for std_label in std_label_set:
                        standard_category_count_list[level][std_label] += 1
                for level, pred_label_set in predict_label_map.items():
                    for pred_label in pred_label_set:
                        predict_category_count_list[level][pred_label] += 1
                for level, std_label_set in standard_label_map.items():
                    for std_label in std_label_set:
                        if level in predict_label_map:
                            for pred_label in predict_label_map[level]:
                                confusion_matrix_list[level][std_label][pred_label] += 1
                                if std_label == pred_label:
                                    right_category_count_list[level][pred_label] += 1

            line_count += 1
        debug_file.close()
        precision_list = []
        recall_list = []
        fscore_list = []
        precision_dict, recall_dict, fscore_dict = \
            self.calculate_level_performance(
                id_to_label_maps[0], right_category_count_list[0],
                predict_category_count_list[0], standard_category_count_list[0],
                exclude_method="start")

        precision_list.append(precision_dict)
        recall_list.append(recall_dict)
        fscore_list.append(fscore_dict)

        for i in range(1, depth + 1):
            precision_dict, recall_dict, fscore_dict = \
                self.calculate_level_performance(
                    id_to_label_maps[i], right_category_count_list[i],
                    predict_category_count_list[i],
                    standard_category_count_list[i], other_text)
            precision_list.append(precision_dict)
            recall_list.append(recall_dict)
            fscore_list.append(fscore_dict)

        self.confusion_matrix_list, self.precision_list, self.recall_list,\
            self.fscore_list, self.right_list, self.predict_list,\
            self.standard_list = (
                confusion_matrix_list, precision_list, recall_list, fscore_list,
                right_category_count_list, predict_category_count_list,
                standard_category_count_list)
        return (confusion_matrix_list, precision_list, recall_list, fscore_list,
                right_category_count_list, predict_category_count_list,
                standard_category_count_list)

    @staticmethod
    def save_confusion_matrix(file_name, confusion_matrix):
        """Save confusion matrix
        Args:
            file_name: File to save to.
            confusion_matrix: Confusion Matrix.
        Returns:
        """
        with open(file_name, "w", encoding=cDataset.CHARSET) as cm_file:
            cm_file.write("\t")
            for category_fist in sorted(confusion_matrix.keys()):
                cm_file.write(category_fist + "\t")
            cm_file.write("\n")
            for category_fist in sorted(confusion_matrix.keys()):
                cm_file.write(category_fist + "\t")
                for category_second in sorted(confusion_matrix.keys()):
                    cm_file.write(
                        str(confusion_matrix[category_fist][
                            category_second]) + "\t")
                cm_file.write("\n")

    def save_prf(self, file_name, precision_category, recall_category,
                 fscore_category, right_category, predict_category,
                 standard_category):
        """Save precision, recall, fscore
        Args:
            file_name: File to save to.
            precision_category: Precision dict.
            recall_category: Recall dict.
            fscore_category: Fscore dict.
            right_category: Right dict.
            predict_category: Predict dict.
            standard_category: Standard dict.
        Returns:
        """

        def _format(category):
            """Format evaluation string.
            Args:
                category: Category evaluation to format.
            Returns:
            """
            if category == self.MACRO_AVERAGE:
                return "%s, precision: %f, recall: %f, fscore: %f, " % (
                    category, precision_category[category],
                    recall_category[category], fscore_category[category])
            return "%s, precision: %f, recall: %f, fscore: %f, " \
                   "right_count: %d, predict_count: %d, " \
                   "standard_count: %d" % (
                       category, precision_category[category],
                       recall_category[category], fscore_category[category],
                       right_category[category], predict_category[category],
                       standard_category[category])

        with open(file_name, "w", encoding=cDataset.CHARSET) as prf_file:
            prf_file.write(_format(self.MACRO_AVERAGE) + "\n")
            prf_file.write(_format(self.MICRO_AVERAGE) + "\n")
            prf_file.write("\n")
            for category in precision_category:
                if category != self.MICRO_AVERAGE and \
                        category != self.MACRO_AVERAGE:
                    prf_file.write(_format(category) + "\n")

    def save(self):
        """Save the latest evaluation.
        """
        for i, confusion_matrix in enumerate(self.confusion_matrix_list):
            if i == 0:
                eval_name = "all"
            else:
                eval_name = "level_%s" % i
            self.save_confusion_matrix(
                self.eval_dir + "/" + eval_name + "_confusion_matrix",
                confusion_matrix)
            self.save_prf(
                self.eval_dir + "/" + eval_name + "_prf",
                self.precision_list[i], self.recall_list[i],
                self.fscore_list[i], self.right_list[i],
                self.predict_list[i], self.standard_list[i])
