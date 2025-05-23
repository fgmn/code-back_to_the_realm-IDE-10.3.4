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


class RNDModel(nn.Module):
    def __init__(self, state_shape, input_size=0, output_size=0):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        cnn_layers = [
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(512, 128), 
            nn.LeakyReLU(inplace=True),
        ]

        self.cnn_model1 = nn.Sequential(*cnn_layers)
        self.cnn_model2 = nn.Sequential(*cnn_layers)

        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(state_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        # Target network
        self.target = nn.Sequential(
            nn.Linear(state_shape, 512),
        )

        # target network is not trainable
        for param in self.cnn_model2.parameters():
            param.requires_grad = False
        for param in self.target.parameters():
            param.requires_grad = False

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

    def forward(self, s):
        feature_vec, feature_maps = s[0], s[1]
        feature_maps1 = self.cnn_model1(feature_maps)
        feature_maps2 = self.cnn_model2(feature_maps)

        feature_maps1 = feature_maps1.view(feature_maps1.shape[0], -1)
        feature_maps2 = feature_maps2.view(feature_maps2.shape[0], -1)

        concat_feature1 = torch.concat([feature_vec, feature_maps1], dim=1)
        concat_feature2 = torch.concat([feature_vec, feature_maps2], dim=1)

        predict_feature = self.predictor(concat_feature1)
        target_feature = self.target(concat_feature2)

        return predict_feature, target_feature


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class DuelingNetwork(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        cnn_layer1 = [
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        cnn_layer4 = [
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]
        cnn_flatten = [nn.Flatten(), nn.Linear(512, 128), nn.ReLU(inplace=True)]

        self.cnn_layer = cnn_layer1 + cnn_layer2 + cnn_layer3 + cnn_layer4 + cnn_flatten
        self.cnn_model = nn.Sequential(*self.cnn_layer)
        
        # 将价值函数和优势函数的分开建模
        self.value_head = nn.Sequential(
            nn.Linear(np.prod(state_shape), 256), 
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(np.prod(state_shape), 256), 
            nn.ReLU(inplace=True), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape))
        )

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

    def forward(self, s, state=None, info=None):
        feature_vec, feature_maps = s[0], s[1]
        feature_maps = self.cnn_model(feature_maps)

        feature_maps = feature_maps.view(feature_maps.shape[0], -1)

        concat_feature = torch.concat([feature_vec, feature_maps], dim=1)

        value = self.value_head(concat_feature)
        advantage = self.advantage_head(concat_feature)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)

        return logits, state


class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        cnn_layer1 = [
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]#压缩长宽的同时增加通道数
        cnn_layer4 = [
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        ]#直接压缩通道维
        cnn_flatten = [nn.Flatten(), nn.Linear(512, 128), nn.ReLU(inplace=True)]

        self.cnn_layer = cnn_layer1 + cnn_layer2 + cnn_layer3 + cnn_layer4 + cnn_flatten
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
