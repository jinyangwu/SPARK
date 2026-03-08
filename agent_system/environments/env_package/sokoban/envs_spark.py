# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import os
import sys
import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from agent_system.environments.env_package.sokoban.sokoban import SokobanEnv

def worker_func(remote, mode, env_kwargs):
    """
    Sokoban 工作进程核心循环：
    1. 在子进程中初始化环境。
    2. 通过 Pipe 接收指令并执行。
    """
    # 初始化环境
    env = SokobanEnv(mode, **env_kwargs)
    action_history = []
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == 'step':
            action = data
            obs, reward, done, info = env.step(action)
            action_history.append(action)
            remote.send((obs, reward, done, info))

        elif cmd == 'reset':
            # data 可以是指定的 seed，如果没有则使用初始传入或随机 seed
            seed_for_reset = data
            obs, info = env.reset(seed=seed_for_reset)
            action_history = []
            remote.send((obs, info))

        elif cmd == 'render':
            mode_for_render = data
            rendered = env.render(mode=mode_for_render)
            remote.send(rendered)

        elif cmd == 'restore_to_state':
            # 逻辑：先重置，再重放动作历史
            replay_actions = data
            action_history = []
            
            for act in replay_actions:
                obs, reward, done, info = env.step(act)
                action_history.append(act)
            
            remote.send((obs, info))

        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")

class SokobanMultiProcessEnv(gym.Env):
    def __init__(self,
                 seed=0, 
                 env_num=1, 
                 group_n=1, 
                 mode='rgb_array',
                 is_train=True,
                 env_kwargs=None):
        super().__init__()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.mode = mode
        
        if env_kwargs is None:
            env_kwargs = {}
        
        # 预生成每个进程的基础 Seed
        np.random.seed(seed)

        self.parent_remotes = []
        self.workers = []

        # 处理多进程上下文
        ctx = mp.get_context('spawn') if sys.platform.startswith("win") else mp.get_context('fork')

        for i in range(self.num_processes):
            parent_remote, child_remote = mp.Pipe()
            worker = ctx.Process(
                target=worker_func,
                args=(child_remote, self.mode, env_kwargs),
                daemon = True
            )
            worker.start()
            self.workers.append(worker)
            child_remote.close()
            
            self.parent_remotes.append(parent_remote)

    def step(self, actions):
        """并行执行 Step"""
        assert len(actions) == self.num_processes
        
        for remote, action in zip(self.parent_remotes, actions):
            remote.send(('step', action))

        results = [remote.recv() for remote in self.parent_remotes]
        
        obs_list, reward_list, done_list, info_list = zip(*results)
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    def reset(self):
        """并行执行 Reset"""
        # 每次 reset 生成新的 seed (保持 group_n 内一致)
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)
        
        seeds = np.repeat(seeds, self.group_n).tolist()

        for i, remote in enumerate(self.parent_remotes):
            remote.send(('reset', seeds[i]))

        results = [remote.recv() for remote in self.parent_remotes]
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def step_with_mask(self, actions: list[str], active_mask: list[int]):
        """
        Executes steps only for the environments specified in active_mask.
        Parallel execution ensures we send all commands first, then wait for results.
        """
        if len(actions) != len(active_mask):
            raise ValueError(
                f'Expected {len(active_mask)} actions for the active mask, got {len(actions)}'
            )

        # 1. Send 'step' command only to active environments
        for env_idx, action in zip(active_mask, actions):
            self.parent_remotes[env_idx].send(('step', action))

        obs_list, reward_list, done_list, info_list = [], [], [], []

        # 2. Receive results only from active environments
        for env_idx in active_mask:
            obs, reward, done, info = self.parent_remotes[env_idx].recv()
            
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            
            # Update the tracked available actions for this specific environment index

        # Return signature matches self.step(): (obs, valid_ids, reward, done, info)
        return obs_list, reward_list, done_list, info_list
    
    def render(self, mode='rgb_array', env_idx=None):
        """并行或指定索引渲染"""
        if env_idx is not None:
            self.parent_remotes[env_idx].send(('render', mode))
            return self.parent_remotes[env_idx].recv()
        else:
            for remote in self.parent_remotes:
                remote.send(('render', mode))
            return [remote.recv() for remote in self.parent_remotes]

    def restore(self, maps: list):
        """
        根据
        maps: [{"parent_idx": int, "child_idx": int, "action_history": list[str]}]
        """
        # 1. 发送所有
        for map_info in maps:
            parent_idx = map_info['parent_idx']
            child_idx = map_info['child_idx']
            action_history = map_info['action_history']
            
            # 关键：从父节点复制
            # self.prev_available_actions[child_idx] = self.prev_available_actions[parent_idx]
            # self.prev_possible_actions[child_idx] = self.prev_possible_actions[parent_idx]
            
            # 向
            remote = self.parent_remotes[child_idx]
            remote.send(('restore_to_state', action_history))

        # 2. 等待所有
        # 这确保了所有
        for map_info in maps:
            child_idx = map_info['child_idx']
            remote = self.parent_remotes[child_idx]
            obs, info = remote.recv()

    def close(self):
        """关闭所有进程"""
        for remote in self.parent_remotes:
            try:
                remote.send(('close', None))
            except:
                pass
        for worker in self.workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()

    def __del__(self):
        self.close()

def build_sokoban_envs(seed=0, env_num=1, group_n=1, mode='rgb_array', is_train=True, env_kwargs=None):
    return SokobanMultiProcessEnv(seed, env_num, group_n, mode, is_train, env_kwargs=env_kwargs)