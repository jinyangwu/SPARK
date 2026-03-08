import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import sys
import os
import time
import random
from typing import Union
from itertools import product
import traceback  # <--- 导入 traceback

tasks = ['boil', 'change-the-state-of-matter-of', 'chemistry-mix', 'chemistry-mix-paint-secondary-color', 'chemistry-mix-paint-tertiary-color', 'find-animal', 'find-living-thing', 'find-non-living-thing', 'find-plant', 'freeze', 'grow-fruit', 'grow-plant', 'identify-life-stages-1', 'identify-life-stages-2', 'inclined-plane-determine-angle', 'inclined-plane-friction-named-surfaces', 'inclined-plane-friction-unnamed-surfaces', 'lifespan-longest-lived', 'lifespan-longest-lived-then-shortest-lived', 'lifespan-shortest-lived', 'measure-melting-point-known-substance', 'measure-melting-point-unknown-substance', 'melt', 'mendelian-genetics-known-plant', 'mendelian-genetics-unknown-plant', 'power-component', 'power-component-renewable-vs-nonrenewable-energy', 'test-conductivity', 'test-conductivity-of-unknown-substances', 'use-thermometer']

def compute_reward(info, multi_modal=False):
    reward = 10.0 * float(info['won'])
    return reward

def _worker(remote, seed, task_nums, simplifications_preset, env_step_limit, jar_path, split=None, variations_idx=None):
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
    taskNames = env.get_task_names()
    random.seed(seed)
    task_id, task_variation = random.choice(variations_idx)
    prev_score = 0
    
    # --- 新增：用于 'restore' 的状态追踪 ---
    # action_history = []
    current_task_name = None
    current_task_variation = None
    current_task_id = -1
    # ------------------------------------

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            action = data
            observation, reward, done, info = env.step(action)
            
            # --- 新增：追踪动作历史 ---
            # action_history.append(action)
            # -------------------------
            
            valid_actions = env.get_possible_actions()
            valid_objs = env.get_possible_objects()
            valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
            info['available_actions'] = valid_action_strs
            info['observation_text'] = observation
            info["possible_actions"] = env.get_valid_action_object_combinations()
            info['score'] = info.get('score', 0.0)
            info['task_score'] = info['score']
            isCompleted = done
            prev_score = info['score']
            info["won"] = isCompleted and info["score"] > 0
            reward = compute_reward(info)
            remote.send((observation, reward, isCompleted, info))
        
        elif cmd == 'reset':
            try:
                if data is None:
                    task_id, task_variation = random.choice(variations_idx)
                    task_num = task_id
                    taskName = taskNames[task_num]
                else:
                    variation_idx = data # Note: This logic seems incomplete from original, but we follow it
                    # # HACK: Assuming data is (task_id, task_variation) if not None
                    # task_id, task_variation = variation_idx
                    # task_num = task_id
                    # taskName = taskNames[task_num]

                # env.close()
                # time.sleep(1)
                # env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)

                simplification_str = simplifications_preset if simplifications_preset else ""
                env.clear_run_histories()
                # env.close()
                env.load(taskName, task_variation, simplification_str)
                # time.sleep(0.1)  # 确保环境正确加载
                observation, info = env.reset()
                
                # --- 新增：存储当前任务信息并重置历史 ---
                current_task_name = taskName
                current_task_variation = task_variation
                current_task_id = task_num
                # action_history = []
                # ---------------------------------------

                task_description = env.get_task_description()
                # print(observation, info, task_description)
                info['task_description'] = task_description
                valid_actions = env.get_possible_actions()
                valid_objs = env.get_possible_objects()
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                info["possible_actions"] = env.get_valid_action_object_combinations()
                info['won'] = False
                info['task_num'] = task_num
                prev_score = 0
                remote.send((observation, info))

            except Exception as e:
                # --- 捕获到 reset 期间的任何错误 ---
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
                print(f"WORKER CRASHED during RESET (Seed: {seed}): {e}", file=sys.stderr)
                # 打印完整的堆栈跟踪
                traceback.print_exc(file=sys.stderr)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
                sys.stderr.flush() # 确保信息被立即打印出来

        # --- 新增：'restore_to_state' 命令处理 ---
        elif cmd == 'restore_to_state':
            replay_actions = data
            for action in replay_actions:
                observation, reward, done, info = env.step(action)
                # action_history.append(action)
                info['score'] = info.get('score', 0.0)
                prev_score = info['score']

            # 3. 发送重放后的最终状态
            task_description = env.get_task_description()
            info['task_description'] = task_description
            valid_actions = env.get_possible_actions()
            valid_objs = env.get_possible_objects()
            valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
            info['available_actions'] = valid_action_strs
            info['observation_text'] = observation
            info["possible_actions"] = env.get_valid_action_object_combinations()
            info['won'] = done and info["score"] > 0
            info['task_num'] = current_task_id
            
            remote.send((observation, info))
        # -----------------------------------------

        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError(f"Unknown command sent to worker: {cmd}")

class SciWorldMultiProcessEnv(gym.Env):
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        task_nums: list = [1], 
        split: str = "train", 
        simplifications_preset: str = "", 
        env_step_limit: int = 100,
        jar_path: str = None,
        variations_idx: list = None  
    ) -> None:
        super().__init__()
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.split = split
        self.task_nums = task_nums
        self.variations_idx = variations_idx
        self.simplifications_preset = simplifications_preset
        self.env_step_limit = env_step_limit
        self.jar_path = jar_path
        random.seed(seed)
        self._rng = np.random.RandomState(seed)
        self._parent_remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []
        
        # --- 修改：确保使用 'spawn' 或 'fork' (在Linux/Mac上 'fork' 通常更快) ---
        if sys.platform.startswith("win"):
            ctx = mp.get_context('spawn')
        else:
            ctx = mp.get_context('fork') # 'fork' 效率更高
        # -----------------------------------------------------------

        for i in range(self.num_processes):
            parent_remote, child_remote = ctx.Pipe()
            seed_i = seed + i
            worker = ctx.Process(
                target=_worker,
                args=(child_remote, seed_i, self.task_nums, self.simplifications_preset, 
                      self.env_step_limit, self.jar_path, self.split, self.variations_idx),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
            self._parent_remotes.append(parent_remote)
            child_remote.close()
        self.prev_available_actions = [[] for _ in range(self.num_processes)]
        self.prev_possible_actions = [[] for _ in range(self.num_processes)]

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )
        for remote, action in zip(self._parent_remotes, actions):
            remote.send(('step', action))
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, reward, done, info = remote.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]
        return obs_list, None, reward_list, done_list, info_list

    # --- 新增：step_with_mask 方法 ---
    def step_with_mask(self, actions: list[str], active_mask: list[int]):
        """
        只对 active_mask 中指定的
        """
        if len(actions) != len(active_mask):
            raise ValueError(
                f'The num of actions ({len(actions)}) must be equal to the num of active envs ({len(active_mask)})'
            )

        # 1. 仅向活动
        for i, idx in enumerate(active_mask):
            remote = self._parent_remotes[idx]
            remote.send(('step', actions[i]))

        # 2. 仅从活动
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, idx in enumerate(active_mask):
            remote = self._parent_remotes[idx]
            obs, reward, done, info = remote.recv()
            
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            
            # 更新特定
            self.prev_available_actions[idx] = info['available_actions']
            self.prev_possible_actions[idx] = info["possible_actions"]
            
        return obs_list, None, reward_list, done_list, info_list
    # ----------------------------------

    def reset(self):
        variations = [None for _ in range(self.num_processes)]
        for remote, variation in zip(self._parent_remotes, variations):
            remote.send(('reset', variation))
        obs_list, info_list = [], []
        for i, remote in enumerate(self._parent_remotes):
            try:
                obs, info = remote.recv()
                obs_list.append(obs)
                info_list.append(info)
                self.prev_available_actions[i] = info['available_actions']
                self.prev_possible_actions[i] = info["possible_actions"]
            except EOFError:
                # 关键：捕获错误并指出是哪个进程失败了
                print(f"\n!!! FATAL: Environment Worker #{i} FAILED before sending reset results !!!", file=sys.stderr)
                print("This worker likely crashed during initialization (ScienceWorldEnv/Java issue) or previously due to an unhandled exception.", file=sys.stderr)
                # 强制退出，因为环境状态已损坏
                self.close()
                raise # 重新抛出错误，但现在有更多诊断信息
        return obs_list, info_list

    # --- 新增：restore 方法 ---
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
            self.prev_available_actions[child_idx] = self.prev_available_actions[parent_idx]
            self.prev_possible_actions[child_idx] = self.prev_possible_actions[parent_idx]
            
            # 向
            remote = self._parent_remotes[child_idx]
            remote.send(('restore_to_state', action_history))

        # 2. 等待所有
        # 这确保了所有
        for map_info in maps:
            child_idx = map_info['child_idx']
            remote = self._parent_remotes[child_idx]
            obs, info = remote.recv()
            
    # ----------------------------

    @property
    def get_available_actions(self):
        return self.prev_available_actions

    @property
    def get_admissible_commands(self):
        return self.prev_available_actions

    @property
    def get_possible_actions(self):
        return self.prev_possible_actions

    def close(self):
        if getattr(self, '_closed', False):
            return
        
        # --- 修改：使用 try-except 增加
        for remote in self._parent_remotes:
            try:
                remote.send(('close', None))
            except BrokenPipeError:
                pass # 
                
        for worker in self._workers:
            worker.join(timeout=5.0) # 
            if worker.is_alive():
                worker.terminate() # 
        # ---------------------------------
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except:
            pass # 

def build_sciworld_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    task_nums: Union[int, list] = 1, 
    split: str = "train", 
    simplifications_preset: str = "",
    env_step_limit: int = 100,
    jar_path: str = None,
    variations_idx: list = None
):
    return SciWorldMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        task_nums=task_nums,
        split=split,
        simplifications_preset=simplifications_preset,
        env_step_limit=env_step_limit,
        jar_path=jar_path,
        variations_idx=variations_idx
    )