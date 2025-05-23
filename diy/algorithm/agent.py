#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
from diy.model.model import Model, DuelingNetwork, RNDModel, \
                            RunningMeanStd, update_mean_var_count_from_moments
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
        self.decay_rate = Config.LR_DECAY
        self.rnd_update_prop = Config.RND_UPDATE_PROP
        self.int_rew_coef = Config.INT_REW_COEF
        self.ext_rew_coef = Config.EXT_REW_COEF
        self.rnd_warmup_steps = Config.RND_WARMUP_STEPS

        self.device = device
        # self.model = Model(
        #     state_shape=self.obs_shape,
        #     action_shape=self.act_shape,
        #     softmax=False,
        # )
        self.model = DuelingNetwork(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
        )
        self.model.to(self.device)
        self.rnd_model = RNDModel(
            state_shape=self.obs_shape,
        ).to(self.device)
        
        combined_parameters = list(self.model.parameters()) + list(self.rnd_model.predictor.parameters())
        self.optim = torch.optim.Adam(combined_parameters, lr=self.lr)
        self.target_model = deepcopy(self.model)

        # 在线白化模块
        self.reward_rms = RunningMeanStd()
        self.vec_rms = RunningMeanStd(shape=(self.obs_split[0],))
        self.map_rms = RunningMeanStd(shape=self.obs_split[1])

        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0

        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

    def linear_schedule(self, step):
        # 学习率线性衰减
        self.lr = max(1e-5, self.lr - self.decay_rate)#5e4*decay_rate=1e-4
        for param_group in self.optim.param_groups:
            param_group["lr"] = self.lr
        # print(f"step: {step}, lr: {self.lr}")

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
        # legal_act = (
        #     torch.cat(
        #         (
        #             legal_act[:, 0].unsqueeze(1).expand(batch, self.direction_space),
        #             legal_act[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
        #         ),
        #         1,
        #     )
        #     .bool()
        #     .to(self.device)
        # )
        legal_act = legal_act.bool().to(self.device)
        model = self.model
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
                feature = [
                    self.__convert_to_tensor(feature_vec),
                    self.__convert_to_tensor(feature_map).view(batch, *self.obs_split[1]),
                ]
                logits, _ = model(feature, state=None)
                logits = logits.masked_fill(~legal_act, float(torch.min(logits)))
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]
    
    def get_int_reward(self, list_obs_data):
        batch = len(list_obs_data)
        feature_vec = [obs_data.feature[: self.obs_split[0]] for obs_data in list_obs_data]
        feature_map = [obs_data.feature[self.obs_split[0] :] for obs_data in list_obs_data]

        feature_vec = self.__convert_to_tensor(feature_vec)
        feature_map = self.__convert_to_tensor(feature_map).view(batch, *self.obs_split[1])
        # 白化观测
        feature_vec = (
            (
            feature_vec - torch.from_numpy(self.vec_rms.mean).to(self.device)
            ) / torch.sqrt(torch.from_numpy(self.vec_rms.var).to(self.device))
        ).clamp(-5, 5).float()
        feature_map = (
            (feature_map - torch.from_numpy(self.map_rms.mean).to(self.device))
            / torch.sqrt(torch.from_numpy(self.map_rms.var).to(self.device))
        ).clamp(-5, 5).float()

        feature = [feature_vec, feature_map]
        
        predict_feature, target_feature = self.rnd_model(feature)
        curiosity_rewards = ((predict_feature - target_feature).pow(2).sum(dim=1) / 2).detach()
        # 从buffer中取出再白化
        # curiosity_rewards /= torch.sqrt(torch.tensor(self.reward_rms.var, device=self.device))
        # 必须打包成list返回
        return curiosity_rewards.cpu().numpy().tolist()

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=True)

    @learn_wrapper
    def learn(self, list_sample_data):
        # 线性衰减学习率
        self.linear_schedule(self.train_step)
        t_data = list_sample_data
        batch = len(t_data)

        # [b, d]
        # obs为t时刻的观测
        # _obs为t+1时刻的观测
        batch_feature_vec = [frame.obs[: self.obs_split[0]] for frame in t_data]
        batch_feature_map = [frame.obs[self.obs_split[0] :] for frame in t_data]
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)

        _batch_obs_legal = torch.tensor(np.array([frame._obs_legal for frame in t_data]))
        # _batch_obs_legal = (
        #     torch.cat(
        #         (
        #             _batch_obs_legal[:, 0].unsqueeze(1).expand(batch, self.direction_space),
        #             _batch_obs_legal[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
        #         ),
        #         1,
        #     )
        #     .bool()
        #     .to(self.device)
        # )
        _batch_obs_legal = _batch_obs_legal.bool().to(self.device)

        int_rew = torch.tensor(np.array([frame.int_rew for frame in t_data]), device=self.device)
        ext_rew = torch.tensor(np.array([frame.ext_rew for frame in t_data]), device=self.device)
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

        # print("batch_feature_vec shape:", np.array(batch_feature_vec).shape)
        # print("batch_feature_map shape:", np.array(batch_feature_map).shape)
        # print("batch_feature[0] shape:", batch_feature[0].shape)
        # print("batch_feature[1] shape:", batch_feature[1].shape)
        # batch_feature_vec shape: (2, 404)
        # batch_feature_map shape: (2, 10404)
        # batch_feature[0] shape: torch.Size([2, 404])
        # batch_feature[1] shape: torch.Size([2, 4, 51, 51])


        # 更新在线白化模块
        self.reward_rms.update(int_rew.cpu().numpy())           # [B]
        self.vec_rms.update(_batch_feature[0].cpu().numpy())    # [B, 404]
        self.map_rms.update(_batch_feature[1].cpu().numpy())    # [B, 4, 51, 51]
        
        # 预热rms
        if self.rnd_warmup_steps > 0:
            self.rnd_warmup_steps -= 1
            return

        # 白化内在奖励(只除以方差，保证非负)
        int_rew /= torch.sqrt(torch.tensor(self.reward_rms.var, device=self.device))
        
        rew = int_rew * self.int_rew_coef + ext_rew * self.ext_rew_coef
        
        # 白化观测
        rnd_batch_feature_vec = (
            (_batch_feature[0] - torch.from_numpy(self.vec_rms.mean).to(self.device))
            / torch.sqrt(torch.from_numpy(self.vec_rms.var).to(self.device))
        ).clamp(-5, 5).float()
        rnd_batch_feature_map = (
            (_batch_feature[1] - torch.from_numpy(self.map_rms.mean).to(self.device))
            / torch.sqrt(torch.from_numpy(self.map_rms.var).to(self.device))
        ).clamp(-5, 5).float()
        rnd_batch_feature = [rnd_batch_feature_vec, rnd_batch_feature_map]

        predict_feature, target_feature = self.rnd_model(rnd_batch_feature) # [B, D]
        # 计算RND前向损失
        forward_loss = F.mse_loss(
            predict_feature, target_feature.detach(), reduction="none"
        ).mean(-1)  # D维度上求均值
        # 给RND predictor的前向损失加上随机mask，防止过快收敛 --> 内在奖励消失
        mask = torch.rand(len(forward_loss), device=self.device)
        mask = (mask < self.rnd_update_prop).type(torch.float32).to(self.device)
        rnd_loss = (forward_loss * mask).sum() / torch.max(
            mask.sum(), torch.tensor(1.0, device=self.device)
        )

        
        q_network = getattr(self, "model")
        target_network = getattr(self, "target_model")
        target_network.eval()
        with torch.no_grad():
            next_q_values, _ = target_network(_batch_feature, state=None)
            next_q_values = next_q_values.masked_fill(~_batch_obs_legal, float(torch.min(next_q_values)))
            
            # Double Q-Learning
            next_q_online, _ = q_network(_batch_feature, state=None)    # [B, num_actions]
            next_q_online = next_q_online.masked_fill(~_batch_obs_legal, float(torch.min(next_q_online)))
            best_actions = torch.argmax(next_q_online, dim=1)   # [B]
            q_max = next_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1).detach()

        q_network.train()
        target_q = rew + self._gamma * q_max * not_done
        logits, h = q_network(batch_feature, state=None)
        loss = torch.square(target_q - logits.gather(1, batch_action).view(-1)).mean()
        loss += rnd_loss
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
        self.optim.step()


        # model = getattr(self, "target_model")
        # model.eval()

        # with torch.no_grad():
        #     q, h = model(_batch_feature, state=None)
        #     q = q.masked_fill(~_batch_obs_legal, float(torch.min(q)))
        #     q_max = q.max(dim=1).values.detach()

        # target_q = rew + self._gamma * q_max * not_done

        # self.optim.zero_grad()

        # model = getattr(self, "model")
        # model.train()
        # logits, h = model(batch_feature, state=None)
        # # logits: [batch_size, num_actions]
        # # logits.gather(1, batch_action): [batch_size, 1]
        # # logits.gather(1, batch_action).view(-1): [batch_size]
        # loss = torch.square(target_q - logits.gather(1, batch_action).view(-1)).mean()
        # loss.backward()
        # model_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # self.optim.step()

        self.train_step += 1

        # Update the target network
        # 更新target网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_q()

        value_loss = loss.detach().item()
        q_value = target_q.mean().detach().item()
        reward = rew.mean().detach().item()
        ext_rew = ext_rew.mean().detach().item()
        int_rew = int_rew.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
                "diy_1": model_grad_norm,
                "diy_2": self.lr,
                "diy_3": ext_rew,
                "diy_4": int_rew,
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

        rnd_model_file_path = f"{path}/rnd_model.ckpt-{str(id)}.pkl"
        rnd_model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.rnd_model.state_dict().items()}
        torch.save(rnd_model_state_dict_cpu, rnd_model_file_path)
        self.logger.info(f"save rnd model {rnd_model_file_path} successfully")

        # 保存三个 RunningMeanStd 的状态
        rms_states = {
            "reward": {
                "mean": self.reward_rms.mean,
                "var": self.reward_rms.var,
                "count": self.reward_rms.count,
            },
            "vec": {
                "mean": self.vec_rms.mean,
                "var": self.vec_rms.var,
                "count": self.vec_rms.count,
            },
            "map": {
                "mean": self.map_rms.mean,
                "var": self.map_rms.var,
                "count": self.map_rms.count,
            },
        }
        rms_file_path = f"{path}/rms.ckpt-{str(id)}.pkl"
        torch.save(rms_states, rms_file_path)
        self.logger.info(f"save rms states {rms_file_path} successfully")

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

        rnd_model_file_path = f"{path}/rnd_model.ckpt-{str(id)}.pkl"
        self.rnd_model.load_state_dict(
            torch.load(rnd_model_file_path, map_location=self.device),
        )
        self.logger.info(f"load rnd model {rnd_model_file_path} successfully")

        # 加载三个 RunningMeanStd 的状态
        rms_file_path = f"{path}/rms.ckpt-{str(id)}.pkl"
        rms_states = torch.load(rms_file_path, map_location="cpu")
        # 直接赋值 numpy 数组
        self.reward_rms.mean = rms_states["reward"]["mean"]
        self.reward_rms.var = rms_states["reward"]["var"]
        self.reward_rms.count = rms_states["reward"]["count"]

        self.vec_rms.mean = rms_states["vec"]["mean"]
        self.vec_rms.var = rms_states["vec"]["var"]
        self.vec_rms.count = rms_states["vec"]["count"]

        self.map_rms.mean = rms_states["map"]["mean"]
        self.map_rms.var = rms_states["map"]["var"]
        self.map_rms.count = rms_states["map"]["count"]

        self.logger.info(f"load rms states {rms_file_path} successfully")

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
