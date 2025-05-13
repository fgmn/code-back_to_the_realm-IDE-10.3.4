#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
from diy.model.model import Model, NoisyDuelingDistributionalNetwork
from diy.feature.definition import ActData
import numpy as np
from copy import deepcopy
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from diy.config import Config


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.epsilon = Config.EPSILON
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.obs_split = Config.DESC_OBS_SPLIT
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR

        self.device = device
        # self.model = Model(
        #     state_shape=self.obs_shape,
        #     action_shape=self.act_shape,
        #     softmax=False,
        # )
        self.model = NoisyDuelingDistributionalNetwork(
            state_shape=self.obs_shape,
            n_atoms=Config.N_ATOMS,
            v_min=Config.V_MIN,
            v_max=Config.V_MAX,
            action_shape=self.act_shape,
            softmax=False,
        )
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.target_model = deepcopy(self.model)
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0

        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            return torch.tensor(
                np.array(data),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            return torch.tensor(
                data,
                device=self.device,
                dtype=torch.float32,
            )

    # take_action
    def __predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)
        #将一维二维特征分开
        feature_vec = [obs_data.feature[: self.obs_split[0]] for obs_data in list_obs_data]
        feature_map = [obs_data.feature[self.obs_split[0] :] for obs_data in list_obs_data]
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = torch.tensor(np.array(legal_act))
        #8个方向移动+8个方向闪现的action mask
        legal_act = legal_act.bool().to(self.device)
        model = self.model
        if exploit_flag:
            model.train()
            model.reset_noise()
        else:
            model.eval()
        
        # Exploration factor,
        # we want epsilon to decrease as the number of prediction steps increases, until it reaches 0.1
        # 探索因子, 我们希望epsilon随着预测步数越来越小，直到0.1为止
        self.epsilon = max(0.1, self.epsilon - self.predict_count / self.egp)

        with torch.no_grad():
            # epsilon greedy
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                random_action = random_action.masked_fill(~legal_act, 0)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                # 使用Noisy Network自带探索
                feature = [
                    self.__convert_to_tensor(feature_vec),
                    self.__convert_to_tensor(feature_map).view(batch, *self.obs_split[1]),
                ]
                q_dist, _ = model(feature, state=None)
                q_values = (q_dist * model.support).sum(dim=2)
                q_values = q_values.masked_fill(~legal_act, float(torch.min(q_values)))
                act = q_values.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=True)

    @learn_wrapper
    def learn(self, list_sample_data):

        t_data = list_sample_data
        batch = len(t_data)

        # [b, d]
        # obs为t时刻的观测
        # _obs为t+1时刻的观测
        batch_feature_vec = [frame.obs[: self.obs_split[0]] for frame in t_data]
        batch_feature_map = [frame.obs[self.obs_split[0] :] for frame in t_data]
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)

        _batch_obs_legal = torch.tensor(np.array([frame._obs_legal for frame in t_data]))
        _batch_obs_legal = _batch_obs_legal.bool().to(self.device)

        rew = torch.tensor(np.array([frame.rew for frame in t_data]), device=self.device)
        _batch_feature_vec = [frame._obs[: self.obs_split[0]] for frame in t_data]
        _batch_feature_map = [frame._obs[self.obs_split[0] :] for frame in t_data]
        not_done = torch.tensor(np.array([0 if frame.done == 1 else 1 for frame in t_data]), device=self.device)

        batch_feature = [
            self.__convert_to_tensor(batch_feature_vec),
            self.__convert_to_tensor(batch_feature_map).view(batch, *self.obs_split[1]),
        ]
        _batch_feature = [
            self.__convert_to_tensor(_batch_feature_vec),
            self.__convert_to_tensor(_batch_feature_map).view(batch, *self.obs_split[1]),
        ]

        model = getattr(self, "model")
        target_model = getattr(self, "target_model")

        # 重新采样噪声
        model.reset_noise()
        target_model.reset_noise()

        model.eval()
        target_model.eval()
        with torch.no_grad():
            next_dist, _ = target_model(_batch_feature, state=None)
            support = target_model.support  # [n_atoms]
            # next_q_values = torch.sum(next_dist * support, dim=2)
            # next_q_values = next_q_values.masked_fill(~_batch_obs_legal, float(torch.min(next_q_values)))

            # double q-learning
            next_dist_online, _ = model(_batch_feature, state=None)  # [B, num_actions, n_atoms]
            next_q_online = torch.sum(next_dist_online * support, dim=2)  # [B, num_actions]
            next_q_online = next_q_online.masked_fill(~_batch_obs_legal, float(torch.min(next_q_online)))
            best_actions = torch.argmax(next_q_online, dim=1)  # [B]
            # pmfs=Probability Mass Functions概率质量函数
            next_pmfs = next_dist[torch.arange(batch), best_actions]    # [B, n_atoms]

            # 由于无法修改采样器的实现，这里只能采用1-step TD
            next_atoms = rew.view(-1, 1) + self._gamma * support * not_done.view(-1, 1)     # [B, n_atoms]
            tz = next_atoms.clamp(model.v_min, model.v_max)

            # projection
            delta_z = model.delta_z
            b = (tz - model.v_min) / delta_z  # shape: [B, n_atoms]
            l = b.floor().clamp(0, Config.N_ATOMS - 1)
            u = b.ceil().clamp(0, Config.N_ATOMS - 1)

            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
            d_m_l = (u.float() + (l == b).float() - b) * next_pmfs  # [B, n_atoms]
            d_m_u = (b - l) * next_pmfs  # [B, n_atoms]

            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        
        model.train()
        dist, _ = model(batch_feature, state=None)  # [B, num_actions, n_atoms]
        pred_dist = dist.gather(1, batch_action.unsqueeze(-1).expand(-1, -1, Config.N_ATOMS)).squeeze(1)
        log_pred = torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5))

        loss_per_sample = -(target_pmfs * log_pred).sum(dim=1)
        # loss = (loss_per_sample * data.weights.squeeze()).mean()
        # 没有优先经验回放，直接取算数平均
        loss = loss_per_sample.mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.train_step += 1

        # Update the target network
        # 更新target网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_q()

        q_values = (pred_dist * support).sum(dim=1)
        target_q_values = (target_pmfs * support).sum(dim=1)

        value_loss = loss.detach().item()
        q_value = q_values.mean().detach().item()
        target_q_value = target_q_values.mean().detach().item()
        reward = rew.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
                "diy_1": target_q_value,
                "diy_2": 0,
                "diy_3": 0,
                "diy_4": 0,
                "diy_5": 0,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(
            torch.load(model_file_path, map_location=self.device),
        )

        self.logger.info(f"load model {model_file_path} successfully")

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
