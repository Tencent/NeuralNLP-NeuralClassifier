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

#copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.optimizer import required


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr=required, warmup=-1, t_total=-1,
                 schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError(
                "Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(
                "Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(
                    warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError(
                "Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup,
                        t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                device = p.device
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)
                    if 'is_embedding' in group and group['is_embedding']:
                        vocab_size = p.data.size(0)
                        state['b1_correction'] = torch.ones([vocab_size],
                                                            device=device)
                        state['b1_correction'][:] = group['b1']
                        state['b2_correction'] = torch.ones([vocab_size],
                                                            device=device)
                        state['b2_correction'][:] = group['b2']
                        state['ones'] = torch.ones([vocab_size], device=device)
                        state['zeros'] = torch.zeros([vocab_size],
                                                     device=device)

                        state['b1'] = torch.ones([vocab_size], device=device)
                        state['b1'][:] = group['b1']
                        state['b2'] = torch.ones([vocab_size], device=device)
                        state['b2'][:] = group['b2']

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                if 'is_embedding' in group and group['is_embedding']:
                    bias_correction1 = 1 - state['b1_correction']
                    bias_correction2 = 1 - state['b2_correction']
                    step_size = lr_scheduled * bias_correction2.sqrt() / bias_correction1
                    step_size = step_size.unsqueeze(1)
                    lr_scheduled = lr_scheduled * step_size
                    grad_condition = torch.ge(torch.abs(grad).sum(1), 1e-6)

                    update_embedding = torch.where(grad_condition,
                                                   state['ones'],
                                                   state['zeros'])
                    lr_scheduled = lr_scheduled * update_embedding.unsqueeze(-1)
                    beta1_tensor = torch.where(grad_condition, state['ones'],
                                               state['b1'])
                    state['b1_correction'].mul_(beta1_tensor)
                    beta2_tensor = torch.where(grad_condition, state['ones'],
                                               state['b2'])
                    state['b2_correction'].mul_(beta2_tensor)

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1
        return loss

