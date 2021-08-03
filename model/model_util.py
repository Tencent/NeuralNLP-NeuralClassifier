#!usr/bin/env python
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

import codecs as cs
import torch

from util import Type
from model.optimizer import BertAdam

class ActivationType(Type):
    """Standard names for activation
    """
    SIGMOID = 'sigmoid'
    TANH = "tanh"
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    NONE = 'linear'

    @classmethod
    def str(cls):
        return ",".join(
            [cls.SIGMOID, cls.TANH, cls.RELU, cls.LEAKY_RELU, cls.NONE])


class InitType(Type):
    """Standard names for init
    """
    UNIFORM = 'uniform'
    NORMAL = "normal"
    XAVIER_UNIFORM = 'xavier_uniform'
    XAVIER_NORMAL = 'xavier_normal'
    KAIMING_UNIFORM = 'kaiming_uniform'
    KAIMING_NORMAL = 'kaiming_normal'
    ORTHOGONAL = 'orthogonal'

    def str(self):
        return ",".join(
            [self.UNIFORM, self.NORMAL, self.XAVIER_UNIFORM, self.XAVIER_NORMAL,
             self.KAIMING_UNIFORM, self.KAIMING_NORMAL, self.ORTHOGONAL])


class FAN_MODE(Type):
    """Standard names for fan mode
    """
    FAN_IN = 'FAN_IN'
    FAN_OUT = "FAN_OUT"

    def str(self):
        return ",".join([self.FAN_IN, self.FAN_OUT])


def init_tensor(tensor, init_type=InitType.XAVIER_UNIFORM, low=0, high=1,
                mean=0, std=1, activation_type=ActivationType.NONE,
                fan_mode=FAN_MODE.FAN_IN, negative_slope=0):
    """Init torch.Tensor
    Args:
        tensor: Tensor to be initialized.
        init_type: Init type, candidate can be found in InitType.
        low: The lower bound of the uniform distribution,
            useful when init_type is uniform.
        high: The upper bound of the uniform distribution,
            useful when init_type is uniform.
        mean: The mean of the normal distribution,
            useful when init_type is normal.
        std: The standard deviation of the normal distribution,
            useful when init_type is normal.
        activation_type: For xavier and kaiming init,
            coefficient is calculate according the activation_type.
        fan_mode: For kaiming init, fan mode is needed
        negative_slope: For kaiming init,
            coefficient is calculate according the negative_slope.
    Returns:
    """
    if init_type == InitType.UNIFORM:
        return torch.nn.init.uniform_(tensor, a=low, b=high)
    elif init_type == InitType.NORMAL:
        return torch.nn.init.normal_(tensor, mean=mean, std=std)
    elif init_type == InitType.XAVIER_UNIFORM:
        return torch.nn.init.xavier_uniform_(
            tensor, gain=torch.nn.init.calculate_gain(activation_type))
    elif init_type == InitType.XAVIER_NORMAL:
        return torch.nn.init.xavier_normal_(
            tensor, gain=torch.nn.init.calculate_gain(activation_type))
    elif init_type == InitType.KAIMING_UNIFORM:
        return torch.nn.init.kaiming_uniform_(
            tensor, a=negative_slope, mode=fan_mode,
            nonlinearity=activation_type)
    elif init_type == InitType.KAIMING_NORMAL:
        return torch.nn.init.kaiming_normal_(
            tensor, a=negative_slope, mode=fan_mode,
            nonlinearity=activation_type)
    elif init_type == InitType.ORTHOGONAL:
        return torch.nn.init.orthogonal_(
            tensor, gain=torch.nn.init.calculate_gain(activation_type))
    else:
        raise TypeError(
            "Unsupported tensor init type: %s. Supported init type is: %s" % (
                init_type, InitType.str()))


class OptimizerType(Type):
    """Standard names for optimizer
    """
    ADAM = "Adam"
    ADADELTA = "Adadelta"
    BERT_ADAM = "BERTAdam"

    def str(self):
        return ",".join([self.ADAM, self.ADADELTA])


def get_optimizer(config, params):
    params = params.get_parameter_optimizer_dict()
    if config.optimizer.optimizer_type == OptimizerType.ADAM:
        return torch.optim.Adam(lr=config.optimizer.learning_rate,
                                params=params)
    elif config.optimizer.optimizer_type == OptimizerType.ADADELTA:
        return torch.optim.Adadelta(
            lr=config.optimizer.learning_rate,
            rho=config.optimizer.adadelta_decay_rate,
            eps=config.optimizer.adadelta_epsilon,
            params=params)
    elif config.optimizer.optimizer_type == OptimizerType.BERT_ADAM:
        return BertAdam(params,
                        lr=config.optimizer.learning_rate,
                        weight_decay=0, max_grad_norm=-1)
    else:
        raise TypeError(
            "Unsupported tensor optimizer type: %s.Supported optimizer "
            "type is: %s" % (config.optimizer_type, OptimizerType.str()))


def get_hierar_relations(hierar_taxonomy, label_map):
    """ get parent-children relationships from given hierar_taxonomy
        hierar_taxonomy: parent_label \t child_label_0 \t child_label_1 \n
    """
    hierar_relations = {}
    new_label_map = {}
    for label_path, idx in label_map.items():
        for label in label_path.split('--'):
            new_label_map[label] = new_label_map.get(label, idx)
    with cs.open(hierar_taxonomy, "r", "utf8") as f:
        for line in f:
            line_split = line.strip("\n").split("\t")
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label not in new_label_map:
                continue
            parent_label_id = new_label_map[parent_label]
            children_label_ids = [new_label_map[child_label] \
                for child_label in children_label if child_label in new_label_map]
            hierar_relations[parent_label_id] = children_label_ids
    return hierar_relations
