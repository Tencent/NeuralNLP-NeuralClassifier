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

import logging
import random
import sys

import numpy as np
import torch

EPS = 1e-7


class Type(object):
    @classmethod
    def str(cls):
        raise NotImplementedError


class ModeType(Type):
    """Standard names for model modes.
    The following standard keys are defined:
    * `TRAIN`: training mode.
    * `EVAL`: evaluation mode.
    * `PREDICT`: inference mode.
    """
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'infer'

    @classmethod
    def str(cls):
        return ",".join([cls.TRAIN, cls.EVAL, cls.PREDICT])


class Logger(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, config):
        if config.log.log_level == "debug":
            logging_level = logging.DEBUG
        elif config.log.log_level == "info":
            logging_level = logging.INFO
        elif config.log.log_level == "warn":
            logging_level = logging.WARN
        elif config.log.log_level == "error":
            logging_level = logging.ERROR
        else:
            raise TypeError(
                "No logging type named %s, candidate is: info, debug, error")
        logging.basicConfig(filename=config.log.logger_file,
                            level=logging_level,
                            format='%(asctime)s : %(levelname)s  %(message)s',
                            filemode="a",
                            datefmt='%Y-%m-%d %H:%M:%S')

    @staticmethod
    def debug(msg):
        """Log debug message
            msg: Message to log
        """
        logging.debug(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def info(msg):
        """"Log info message
            msg: Message to log
        """
        logging.info(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def warn(msg):
        """Log warn message
            msg: Message to log
        """
        logging.warning(msg)
        sys.stdout.write(msg + "\n")

    @staticmethod
    def error(msg):
        """Log error message
            msg: Message to log
        """
        logging.error(msg)
        sys.stderr.write(msg + "\n")


def pytorch_seed(use_cudnn, seed=2019):
    # Reference: "https://pytorch.org/docs/stable/notes/randomness.html"
    # Pytorch
    torch.manual_seed(seed)
    if use_cudnn:
        # CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # numpy
    np.random.seed(seed)
    # random
    random.seed(seed)
