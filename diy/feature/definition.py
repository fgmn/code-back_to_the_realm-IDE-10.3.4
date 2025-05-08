#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :definition.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from kaiwu_agent.back_to_the_realm.target_dqn.feature_process import (
    one_hot_encoding,
    read_relative_position,
    bump,
)


# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)


def reward_shaping(frame_no, score, terminated, truncated, obs, _obs, env_info, _env_info):
    pass


@attached
def observation_process(raw_obs, env_info=None):
    """
    This function is an important feature processing function, mainly responsible for:
        - Parsing information in the raw data
        - Parsing preprocessed feature data
        - Processing the features and returning the processed feature vector
        - Concatenation of features
        - Annotation of legal actions
    Function inputs:
        - raw_obs: Preprocessed feature data
        - env_info: Environment information returned by the game
    Function outputs:
        - observation: Feature vector
        - legal_action: Annotation of legal actions

    该函数是特征处理的重要函数, 主要负责：
        - 解析原始数据里的信息
        - 解析预处理后的特征数据
        - 对特征进行处理, 并返回处理后的特征向量
        - 特征的拼接
        - 合法动作的标注
    函数的输入：
        - raw_obs: 预处理后的特征数据
        - env_info: 游戏返回的环境信息
    函数的输出：
        - observation: 特征向量
        - legal_action: 合法动作的标注
    """
    pass


@attached
def action_process(act_data):
    pass


@attached
def sample_process(list_game_data):
    pass


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
    pass


@attached
def NumpyData2SampleData(s_data):
    pass
