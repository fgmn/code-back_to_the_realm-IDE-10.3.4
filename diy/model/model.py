#!/usr/bin/env python3
# -*- coding:utf-8 -*-


"""
@Project :back_to_the_realm
@File    :model.py
@Author  :kaiwu
@Date    :2022/11/15 20:57

"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import math

# Noisy Nets
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        # factorized gaussian noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


# ALGO LOGIC: initialize agent here:
class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, state_shape, n_atoms, v_min, v_max, action_shape=0, softmax=False):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = action_shape
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        #CNN
        cnn_layer1 = [
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ]
        max_pool = [nn.MaxPool2d(kernel_size=(2, 2))]
        self.cnn_layer = cnn_layer1 + max_pool + cnn_layer2 + max_pool + cnn_layer3 + max_pool
        self.cnn_model = nn.Sequential(*self.cnn_layer)

        #FC
        self.value_head = nn.Sequential(
            NoisyLinear(state_shape, 512), nn.ReLU(), NoisyLinear(512, n_atoms)
        )

        self.advantage_head = nn.Sequential(
            NoisyLinear(state_shape, 512), nn.ReLU(), NoisyLinear(512, n_atoms * self.n_actions)
        )

        # 初始化
        self.reset_parameters()

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_parameters(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_parameters()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_parameters()
        self.cnn_model.apply(self.init_weights)

    def reset_noise(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, s, state=None, info=None):
        feature_vec, feature_maps = s[0], s[1]
        feature_maps = self.cnn_model(feature_maps)

        feature_maps = feature_maps.view(feature_maps.shape[0], -1)

        concat_feature = torch.concat([feature_vec, feature_maps], dim=1)

        #Dueling networks&Distributional networks
        value = self.value_head(concat_feature).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(concat_feature).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)

        return q_dist, state
        # return q_dist


class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        cnn_layer1 = [
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ]
        max_pool = [nn.MaxPool2d(kernel_size=(2, 2))]
        self.cnn_layer = cnn_layer1 + max_pool + cnn_layer2 + max_pool + cnn_layer3 + max_pool
        self.cnn_model = nn.Sequential(*self.cnn_layer)

        fc_layer1 = [nn.Linear(np.prod(state_shape), 256), nn.ReLU(inplace=True)]
        fc_layer2 = [nn.Linear(256, 128), nn.ReLU(inplace=True)]
        fc_layer3 = [nn.Linear(128, np.prod(action_shape))]

        self.fc_layers = fc_layer1 + fc_layer2

        if action_shape:
            self.fc_layers += fc_layer3
        if softmax:
            self.fc_layers += [nn.Softmax(dim=-1)]

        self.model = nn.Sequential(*self.fc_layers)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Forward inference
    # 前向推理
    def forward(self, s, state=None, info=None):
        feature_vec, feature_maps = s[0], s[1]
        feature_maps = self.cnn_model(feature_maps)

        feature_maps = feature_maps.view(feature_maps.shape[0], -1)

        concat_feature = torch.concat([feature_vec, feature_maps], dim=1)

        logits = self.model(concat_feature)
        return logits, state
