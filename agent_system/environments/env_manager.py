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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory


from copy import deepcopy

### padding to target_size
def pad_with_mask(source_list, active_indices, target_size):
    padded_list = ['[pad]'] * target_size
    for i, idx in enumerate(active_indices):
        padded_list[idx] = source_list[i]
    return padded_list

### getting sublist according to active_indices
def unpad_with_mask(source_list, active_indices):
    return [source_list[idx] for idx in active_indices]
        
### for alfworld
def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

### for alfworld
def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.plannings = []
        self.brainstorms = []

        super().__init__(envs, projection_f, config)
    
    def reset(self):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)
        # RESET PLANNINGS
        self.plannings = ['No plan.' for _ in range(len(text_obs))]
        self.brainstorms = ['No brainstorm.' for _ in range(len(text_obs))]
        self.pure_actions = [[] for _ in range(len(text_obs))]

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids, reasoning_outputs, action_available  = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions, 'react': deepcopy(text_actions)})
        self.pre_text_obs = text_obs

        ### update planning
        plannings = reasoning_outputs['planning']
        for i, plan in enumerate(plannings):
            if plan:
                self.plannings[i] = plan
                
        ### update brainstorms
        brainstorms = reasoning_outputs['explore']
        for i, brainstorm in enumerate(brainstorms):
            if brainstorm:
                self.brainstorms[i] = brainstorm

        ### update pure_actions
        for i, act in enumerate(actions):
            self.pure_actions[i].append(act)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=False)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def step_with_mask(self, text_actions: List[str], active_indices: List[str]):
        admissible_commands = [self.envs.get_admissible_commands[idx] for idx in active_indices]
        actions, valids, reasoning_outputs, action_available = self.projection_f(text_actions, admissible_commands)

        text_obs, image_obs, rewards, dones, infos = self.envs.step_with_mask(actions, active_indices)

        def pad_with_mask(source_list, active_indices):
            padded_list = ['[pad]'] * self.envs.num_processes
            for i, idx in enumerate(active_indices):
                padded_list[idx] = source_list[i]
            return padded_list
            
        # 先补全actions
        self.memory.store({'text_obs': self.pre_text_obs, 'action': pad_with_mask(actions, active_indices), 'react': pad_with_mask(deepcopy(text_actions), active_indices)})

        padded_text_obs = pad_with_mask(text_obs, active_indices)
        self.pre_text_obs = padded_text_obs

        ### update plannings
        plannings = reasoning_outputs['planning']
        for i, idx in enumerate(active_indices):
            plan = plannings[i]
            if plan:
                self.plannings[idx] = plan

        ### update brainstorms
        brainstorms = reasoning_outputs['explore']
        for i, idx in enumerate(active_indices):
            brainstorm = brainstorms[i]
            if brainstorm:
                self.brainstorms[idx] = brainstorm

        ### update pure_actions
        for i, idx in enumerate(active_indices):
            act = actions[i]
            self.pure_actions[idx].append(act)

        padded_full_text_obs = self.build_text_obs(padded_text_obs, self.envs.get_admissible_commands, init=False)
        full_text_obs = [padded_full_text_obs[idx] for idx in active_indices]

        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, [self.gamefile[idx] for idx in active_indices])

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['is_action_available'] = to_numpy(action_available[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def sync_branch_states(self, all_branches: Dict[int, List[int]]):
        maps = []
        for parent_idx, childs_idx in all_branches.items():
            for child_idx in childs_idx:

                # 复制 task 以及 gamefile
                self.tasks[child_idx] = self.tasks[parent_idx]
                self.gamefile[child_idx] = self.gamefile[parent_idx]

                # pre_text_obs
                self.pre_text_obs[child_idx] = self.pre_text_obs[parent_idx]
                # plannings
                self.plannings[child_idx] = self.plannings[parent_idx]
                self.brainstorms[child_idx] = self.brainstorms[parent_idx]

                self.pure_actions[child_idx] = deepcopy(self.pure_actions[parent_idx])
                
                if hasattr(self.memory, '_data'):
                    self.memory._data[child_idx] = deepcopy(self.memory._data[parent_idx])

                action_history = [turn['action'] for turn in self.memory._data[parent_idx]]
                maps.append({"parent_idx": parent_idx, "child_idx": child_idx, "action_history": action_history})
        self.envs.restore(maps)

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """

        if self.config.env.prompt_type == 'react':
            _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_REACT
            _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_REACT
        elif self.config.env.prompt_type == 'spark':
            _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_SPARK
            _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_SPARK
        elif self.config.env.prompt_type == 'rlvmr':
            _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_RLVMR
            _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_RLVMR
        else:
            raise ValueError(f"Unknown prompt type: {self.config.env.prompt_type}")
        
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            if self.config.env.prompt_type == 'rlvmr':
                # use full obs, action history for rlvmr
                memory_contexts, valid_lens = self.memory.fetch_rlvmr(
                    self.config.env.max_steps,
                    obs_key="text_obs",
                    action_key="action",
                    reasoning_key="react")
            else:
                memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                
                obs = _ALFWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    planning = self.plannings[i],
                )
            else:
               
                obs = _ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    planning=self.plannings[i],
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

class SciWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.plannings = []
        super().__init__(envs, projection_f, config)
        print("PROMPT_TYPE:", self.config.env.prompt_type)
    
    def reset(self):
        text_obs, infos = self.envs.reset()
        # import pdb
        # pdb.set_trace()
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task_descriptions(infos)
        # RESET PLANNINGS
        self.plannings = ['No plan.' for _ in range(len(text_obs))]
        self.pure_actions = [[] for _ in range(len(text_obs))]

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos], init=True)
        return {'text': full_text_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids, plannings, action_available  = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions, 'react': text_actions})
        self.pre_text_obs = text_obs

        ### update planning
        for i, plan in enumerate(plannings):
            if plan:
                self.plannings[i] = plan

        ### update pure_actions
        for i, act in enumerate(actions):
            self.pure_actions[i].append(act)
        
        # infos[0]['available_actions']
        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=False)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task_descriptions(self, infos: List[dict]):
        for info in infos:
            if 'task_description' in info:
                self.tasks.append(info['task_description'])
            else:
                self.tasks.append("Unknown task")

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """

        if self.config.env.prompt_type == 'react':
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_REACT
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE_REACT
        elif self.config.env.prompt_type == 'spark':
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_SPARK
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE_SPARK
        elif self.config.env.prompt_type == 'rlvmr':
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_RLVMR
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE_RLVMR
        else:
            raise ValueError(f"Unknown prompt type: {self.config.env.prompt_type}")
        
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            if self.config.env.prompt_type == 'rlvmr':
                # use full obs, action history for rlvmr
                memory_contexts, valid_lens = self.memory.fetch_rlvmr(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action",
                    reasoning_key="react")
            else:
                memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = admissible_actions[i]

            if init or self.config.env.history_length <= 0:
                
                obs = _SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    planning=self.plannings[i]
                )
            else:
               
                obs = _SCIWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    planning=self.plannings[i],
                    pure_actions=self.pure_actions[i]
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = _SCIWORLD_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions,
                        planning=self.plannings[i]
                    )
                

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def step_with_mask(self, text_actions: List[str], active_indices: List[str]):
        admissible_commands = [self.envs.get_admissible_commands[idx] for idx in active_indices]
        actions, valids, plannings, action_available = self.projection_f(text_actions, admissible_commands)

        # SciWorld的step_with_mask也应该返回image_obs，与其常规step方法一致
        text_obs, image_obs, rewards, dones, infos = self.envs.step_with_mask(actions, active_indices)

        def pad_with_mask(source_list, active_indices):
            # 假设 self.envs.num_processes 存在，就像在 AlfWorld 中一样
            padded_list = ['[pad]'] * self.envs.num_processes
            for i, idx in enumerate(active_indices):
                padded_list[idx] = source_list[i]
            return padded_list
            
        # 存储时使用 pad_with_mask 保持 memory 完整
        self.memory.store({'text_obs': self.pre_text_obs, 'action': pad_with_mask(actions, active_indices), 'react': pad_with_mask(text_actions, active_indices)})

        padded_text_obs = pad_with_mask(text_obs, active_indices)
        self.pre_text_obs = padded_text_obs

        ### update plannings
        for i, idx in enumerate(active_indices):
            plan = plannings[i]
            if plan:
                self.plannings[idx] = plan
        ### update pure_actions
        for i, idx in enumerate(active_indices):
            act = actions[i]
            self.pure_actions[idx].append(act)

        # 构建完整的观测，然后只取出激活索引对应的部分
        padded_full_text_obs = self.build_text_obs(padded_text_obs, self.envs.get_admissible_commands, init=False)
        full_text_obs = [padded_full_text_obs[idx] for idx in active_indices]

        # SciWorld 没有 gamefile 逻辑，所以跳过
        # if infos[0].get("extra.gamefile") is None:
        #     infos = set_gamefile(infos, [self.gamefile[idx] for idx in active_indices])

        # add action_valid to infos (参考 AlfWorld，同时添加 is_action_available)
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['is_action_available'] = to_numpy(action_available[i])

        # 返回值也应包含 image_obs
        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def sync_branch_states(self, all_branches: Dict[int, List[int]]):
        # 假设 from copy import deepcopy 已经在文件顶部导入
        maps = []
        for parent_idx, childs_idx in all_branches.items():
            for child_idx in childs_idx:

                # 复制 task
                self.tasks[child_idx] = self.tasks[parent_idx]
                
                # SciWorld 没有 gamefile，跳过
                # self.gamefile[child_idx] = self.gamefile[parent_idx]

                # pre_text_obs
                self.pre_text_obs[child_idx] = self.pre_text_obs[parent_idx]
                # plannings
                self.plannings[child_idx] = self.plannings[parent_idx]
                # pure_actions (需要深拷贝)
                self.pure_actions[child_idx] = deepcopy(self.pure_actions[parent_idx])
                
                # 复制memory状态
                if hasattr(self.memory, '_data'):
                    self.memory._data[child_idx] = deepcopy(self.memory._data[parent_idx])

                action_history = [turn['action'] for turn in self.memory._data[parent_idx]]
                maps.append({"parent_idx": parent_idx, "child_idx": child_idx, "action_history": action_history})
        
        self.envs.restore(maps)
        
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.plannings = []
        super().__init__(envs, projection_f, config)
    
    def reset(self) -> Dict[str, Any]:
        text_obs, infos = self.envs.reset()
        self.tasks = self.extract_task(text_obs)
        text_obs = self.format_obs(text_obs)
        # infos = [None] * self.envs.num_envs
        self.plannings = ['No plan.' for _ in range(len(text_obs))]
        self.pure_actions = [[] for _ in range(len(text_obs))]

        observations = {'text': self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True), 
                        'image': None, 
                        'anchor': text_obs.copy()
                        }
        self.pre_text_obs = text_obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids, plannings, action_available  = self.projection_f(text_actions, self.envs.get_admissible_commands)
        next_obs, image_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions, 'react': text_actions})
        self.pre_text_obs = next_obs

        ### update planning
        for i, plan in enumerate(plannings):
            if plan:
                self.plannings[i] = plan

        ### update pure_actions
        for i, act in enumerate(actions):
            self.pure_actions[i].append(act)
        
        next_observations = {
            'text': self.build_text_obs(next_obs, self.envs.get_admissible_commands, init=False),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def step_with_mask(self, text_actions: List[str], active_indices: List[str]):
        admissible_commands = [self.envs.get_admissible_commands[idx] for idx in active_indices]
        actions, valids, plannings, action_available = self.projection_f(text_actions, admissible_commands)

        text_obs, image_obs, rewards, dones, infos = self.envs.step_with_mask(actions, active_indices)

        def pad_with_mask(source_list, active_indices):
            # 假设 self.envs.num_processes 存在，就像在 AlfWorld 中一样
            padded_list = ['[pad]'] * self.envs.num_processes
            for i, idx in enumerate(active_indices):
                padded_list[idx] = source_list[i]
            return padded_list
            
        # 存储时使用 pad_with_mask 保持 memory 完整
        self.memory.store({'text_obs': self.pre_text_obs, 'action': pad_with_mask(actions, active_indices), 'react': pad_with_mask(text_actions, active_indices)})

        text_obs = self.format_obs(text_obs)
        padded_text_obs = pad_with_mask(text_obs, active_indices)
        self.pre_text_obs = padded_text_obs

        ### update plannings
        for i, idx in enumerate(active_indices):
            plan = plannings[i]
            if plan:
                self.plannings[idx] = plan
        ### update pure_actions
        for i, idx in enumerate(active_indices):
            act = actions[i]
            self.pure_actions[idx].append(act)

        # 构建完整的观测，然后只取出激活索引对应的部分
        padded_full_text_obs = self.build_text_obs(padded_text_obs, self.envs.get_admissible_commands, init=False)
        full_text_obs = [padded_full_text_obs[idx] for idx in active_indices]

        # add action_valid to infos (参考 AlfWorld，同时添加 is_action_available)
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['is_action_available'] = to_numpy(action_available[i])

        # 返回值也应包含 image_obs
        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def sync_branch_states(self, all_branches: Dict[int, List[int]]):
        # 假设 from copy import deepcopy 已经在文件顶部导入
        maps = []
        for parent_idx, childs_idx in all_branches.items():
            for child_idx in childs_idx:

                # 复制 task
                self.tasks[child_idx] = self.tasks[parent_idx]
                
                # SciWorld 没有 gamefile，跳过
                # self.gamefile[child_idx] = self.gamefile[parent_idx]

                # pre_text_obs
                self.pre_text_obs[child_idx] = self.pre_text_obs[parent_idx]
                # plannings
                self.plannings[child_idx] = self.plannings[parent_idx]
                # pure_actions (需要深拷贝)
                self.pure_actions[child_idx] = deepcopy(self.pure_actions[parent_idx])
                
                # 复制memory状态
                if hasattr(self.memory, '_data'):
                    self.memory._data[child_idx] = deepcopy(self.memory._data[parent_idx])

                action_history = [turn['action'] for turn in self.memory._data[parent_idx]]
                maps.append({"parent_idx": parent_idx, "child_idx": child_idx, "action_history": action_history})
        
        self.envs.restore(maps)
        
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"{p}" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
    
    
    def build_text_obs(self, text_obs: List[str], admissible_commands, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """

        if self.config.env.prompt_type == 'react':
            _WEBSHOP_TEMPLATE_NO_HIS = WEBSHOP_TEMPLATE_NO_HIS_REACT
            _WEBSHOP_TEMPLATE = WEBSHOP_TEMPLATE_REACT
        elif self.config.env.prompt_type == 'spark':
            _WEBSHOP_TEMPLATE_NO_HIS = WEBSHOP_TEMPLATE_NO_HIS_SPARK
            _WEBSHOP_TEMPLATE = WEBSHOP_TEMPLATE_SPARK
        elif self.config.env.prompt_type == 'rlvmr':
            _WEBSHOP_TEMPLATE_NO_HIS = WEBSHOP_TEMPLATE_NO_HIS_RLVMR
            _WEBSHOP_TEMPLATE = WEBSHOP_TEMPLATE_RLVMR
        else:
            raise ValueError(f"Unknown prompt type: {self.config.env.prompt_type}")

        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            if self.config.env.prompt_type == 'rlvmr':
                # use full obs, action history for rlvmr
                memory_contexts, valid_lens = self.memory.fetch_rlvmr(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action",
                    reasoning_key="react")
            else:
                memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
            # memory_contexts = post_process_memory(memory_contexts, max_len=20000)
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(admissible_commands[i]) if admissible_commands[i] else {}
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = _WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_available_actions,
                    planning=self.plannings[i],
                )
            else:
                obs = _WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_available_actions,
                    planning=self.plannings[i],

                )
                if len(obs) > 20000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = _WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_available_actions,
                    planning=self.plannings[i],
                )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

from copy import deepcopy
import numpy as np
from typing import List, Dict

class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    
    # Statically define the list of available actions in Sokoban for Prompting
    ADMISSIBLE_ACTIONS = ["up", "down", "left", "right"]

    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        self.plannings = [] # NEW: Used to store planning/thought processes
        super().__init__(envs, projection_f, config)
        print("PROMPT_TYPE:", self.config.env.prompt_type)

    def reset(self):
        obs, infos = self.envs.reset()

        # NEW: Initialize plannings and pure_actions
        self.plannings = ['No plan.' for _ in range(len(infos))]
        self.pure_actions = [[] for _ in range(len(infos))]
        
        if self.is_multi_modal:
            obs_img = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            # Initial construction of observations
            full_text_obs = self.build_text_obs(self.pre_text_obs, init=True)
            observations = {
                'text': full_text_obs,
                'image': obs_img,
                'anchor': obs_img
            }
        else:
            self.pre_text_obs = obs
            full_text_obs = self.build_text_obs(self.pre_text_obs, init=True)
            observations = {
                'text': full_text_obs,
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids, plannings, action_available = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        # NEW: Update plannings
        for i, plan in enumerate(plannings):
            if plan:
                self.plannings[i] = plan
        
        # NEW: Update pure_actions
        for i, act in enumerate(actions):
            self.pure_actions[i].append(self.ACTION_LOOKUP.get(act, "Unknown"))

        # Update validity flags in info
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['is_action_available'] = to_numpy(action_available[i])

        # Store to Memory
        self.memory.store({'text_obs': self.pre_text_obs, 'action': [act for act in actions]})

        if self.is_multi_modal:
            next_obs_img = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            full_text_obs = self.build_text_obs(self.pre_text_obs)
            next_observations = {
                'text': full_text_obs,
                'image': next_obs_img,
                'anchor': next_obs_img
            }
        else:
            self.pre_text_obs = next_obs
            full_text_obs = self.build_text_obs(self.pre_text_obs)
            next_observations = {
                'text': full_text_obs,
                'image': None,
                'anchor': next_obs
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        Build text observations and select templates based on prompt_type.
        """
        # 1. Template selection logic
        if self.config.env.prompt_type == 'react':
            _SOKOBAN_VISUAL_TEMPLATE = SOKOBAN_VISUAL_TEMPLATE_REACT
        elif self.config.env.prompt_type == 'spark':
            _SOKOBAN_VISUAL_TEMPLATE = SOKOBAN_VISUAL_TEMPLATE_SPARK

        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):            
            if init or self.config.env.history_length <= 0:
                # Compatible with original Visual Template logic
                if self.is_multi_modal:
                    obs = _SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = _SOKOBAN_VISUAL_TEMPLATE
                    pass
                    # obs = _TEMPLATE_NO_HIS.format(...)
            else:
                if self.is_multi_modal:
                    obs = _SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = _SOKOBAN_VISUAL_TEMPLATE
                    pass
                    # Use full template including planning and pure_actions
                    # Note: If the {pure_actions} placeholder is not defined in the template, 
                    # format will either ignore extra parameters or throw an error.
                    # Ensure the template definition matches these parameters.
                    # obs = _TEMPLATE.format(...)
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def step_with_mask(self, text_actions: List[str], active_indices: List[int]):
        """
        Step function supporting masking, used for scenarios like Tree Search.
        """
        # Sokoban does not require dynamic admissible commands; passing None/empty is fine.
        # projection_f needs to handle this.
        actions, valids, plannings, action_available = self.projection_f(text_actions)

        # Assuming underlying envs.step_with_mask supports returning image_obs.
        # If the underlying env doesn't support image returns, adaptation is needed here.
        next_obs, rewards, dones, infos = self.envs.step_with_mask(actions, active_indices)
        
        def pad_with_mask(source_list, active_indices):
            padded_list = ['[pad]'] * self.envs.num_processes
            for i, idx in enumerate(active_indices):
                padded_list[idx] = source_list[i]
            return padded_list
        
        # Memory Store
        self.memory.store({
            'text_obs': self.pre_text_obs, 
            'action': pad_with_mask(actions, active_indices)
            # 'planning': pad_with_mask(plannings, active_indices) # If memory needs to store planning
        })

        # Process observation data
        if self.is_multi_modal:
            next_obs_img = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array') # Update text observation cache
        else:
            self.pre_text_obs = pad_with_mask(next_obs, active_indices)
            next_obs_img = None

        # Update Plannings and Pure Actions
        for i, idx in enumerate(active_indices):
            plan = plannings[i]
            if plan:
                self.plannings[idx] = plan
            
            act = actions[i]
            self.pure_actions[idx].append(self.ACTION_LOOKUP.get(act, "Unknown"))

        # Build full observations
        padded_full_text_obs = self.build_text_obs(self.pre_text_obs, init=False)
        # Extract only the active portions
        active_text_obs = [padded_full_text_obs[idx] for idx in active_indices]

        ## Only for active indices, add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['is_action_available'] = to_numpy(action_available[i])

        next_observations = {
            'text': active_text_obs, 
            'image': next_obs_img, 
            'anchor': next_obs if not self.is_multi_modal else next_obs_img
        }
        
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def sync_branch_states(self, all_branches: Dict[int, List[int]]):
        """
        Synchronize branch states, used for backtracking in Tree Search.
        """
        maps = []
        for parent_idx, childs_idx in all_branches.items():
            for child_idx in childs_idx:
                # 1. Copy Task (Even though Sokoban is simple, perform copy)
                # self.tasks[child_idx] = self.tasks[parent_idx]
                
                # 2. Copy pre_text_obs
                self.pre_text_obs[child_idx] = deepcopy(self.pre_text_obs[parent_idx])
                
                # 3. Copy plannings
                self.plannings[child_idx] = self.plannings[parent_idx]
                
                # 4. Copy pure_actions (deep copy)
                self.pure_actions[child_idx] = deepcopy(self.pure_actions[parent_idx])
                
                # 5. Copy Memory
                if hasattr(self.memory, '_data'):
                    self.memory._data[child_idx] = deepcopy(self.memory._data[parent_idx])

                # 6. Prepare restore map for underlying environment
                # Assumes action history in memory is sufficient for reconstruction, 
                # or the underlying env supports snapshot recovery.
                action_history = [turn['action'] for turn in self.memory._data[parent_idx]]
                maps.append({
                    "parent_idx": parent_idx, 
                    "child_idx": child_idx, 
                    "action_history": action_history
                })
        
        # Call the underlying environment's restore method
        self.envs.restore(maps)

class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, **kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # Add text observations to the next_observations dictionary
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos

    def step_with_mask(self, text_actions: List[str], active_indices: List[int]):
        """
        Step function supporting masking, typically used for parallel tree search.
        """
        next_observations, rewards, dones, infos = super().step(text_actions, active_indices)
        
        # Add text observations to the next_observations dictionary
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos

    def sync_branch_states(self, all_branches: Dict[int, List[int]]):
        """
        Synchronize environment states across branches for search-based algorithms.
        """
        self.envs.restore(all_branches)

    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        Constructs text-based observations for the agent based on the environment's current state.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            # Case for 'ezpoints' environments
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                if self.config.env.prompt_type == 'react':
                    obs = GYM_CARDS_EZPOINTS_TEMPLATE_REACT.format(text_formula=text_formula)
                elif self.config.env.prompt_type == 'spark':
                    obs = GYM_CARDS_EZPOINTS_TEMPLATE_SPARK.format(text_formula=text_formula)
            
            # Case for 'points24' (24-point game) environments
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            
            # Case for 'numberline' environments
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            
            # Case for 'blackjack' environments
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            
            postprocess_text_obs.append(obs)
            
        return postprocess_text_obs

class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        
        # Store the current search action and returned information into memory
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def step_with_mask(self, text_actions: List[str], active_indices: List[int]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step_with_mask(actions, active_indices)
        
        ## modified: Handle padding for masked operations to maintain batch consistency
        padded_actions = pad_with_mask(actions, active_indices, len(self.envs))
        padded_next_obs = pad_with_mask(next_obs, active_indices, len(self.envs))
        self.memory.store({
            "search": padded_actions,
            "information": padded_next_obs
        })
        
        ## modified: Unpad the text observations for the active indices
        next_observations = {
            "text": unpad_with_mask(self.build_text_obs(padded_next_obs), active_indices),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def sync_branch_states(self, all_branches: Dict[int, List[int]]):
        """
        Synchronize branch states. Since the Search environment is stateless, 
        we only need to update the memory.
        """
        for parent_idx, childs_idx in all_branches.items():
            for child_idx in childs_idx:
                # Deep copy memory from parent branch to child branches
                self.memory._data[child_idx] = deepcopy(self.memory._data[parent_idx])

        self.envs.sync_branch_states(all_branches)

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False,
    ) -> List[str]:
        """
        Builds the text observation by selecting templates and incorporating memory history.
        """
        postprocess_text_obs: List[str] = []

        # Fetch history from memory if not the initial step and history is enabled
        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        # Template selection logic based on prompt type (ReAct vs Spark)
        if self.config.env.prompt_type == 'react':
            _SEARCH_TEMPLATE_NO_HIS = SEARCH_TEMPLATE_NO_HIS_REACT
            _SEARCH_TEMPLATE = SEARCH_TEMPLATE_REACT
        elif self.config.env.prompt_type == 'spark':
            _SEARCH_TEMPLATE_NO_HIS = SEARCH_TEMPLATE_NO_HIS_SPARK
            _SEARCH_TEMPLATE = SEARCH_TEMPLATE_SPARK

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                # Use simplified template for initialization or when no history is used
                obs_i = _SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                # Use full template including historical context
                obs_i = _SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """
        Post-process a batch to record success rates and data source statistics.
        """
        # Iterate backwards to find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Record success rate specifically for the given data source
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after processing the most recent active step


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    use_api_as_teacher = config.env.get('use_api_as_teacher', False)
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    if "alfworld" in config.env.env_name.lower():
        os.environ["ALFWORLD_DATA"] = config.env.alfworld.data_path
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection_spark
        # if config.env.env_name == 'alfworld/AlfredThorEnv':
        #     alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')

        if config.env.alfworld.generalization_level == 2:
            alf_train_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_train_ood.yaml')
            alf_test_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_test_ood.yaml')
            print("[alfworld] generalization_level 2")
            print(f"[alfworld] use_api_as_teacher: {use_api_as_teacher}")
            _envs = build_alfworld_envs(alf_train_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True)
            if use_api_as_teacher:
                _val_envs = build_alfworld_envs(alf_train_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=True)
            else:
                _val_envs = build_alfworld_envs(alf_test_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, unseen=True)

        elif config.env.alfworld.generalization_level == 1:
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True)
            _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=use_api_as_teacher, unseen=True)
        elif config.env.alfworld.generalization_level == 0:
            print(alf_config_path, config.env.seed, config.data.train_batch_size, group_n)
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True)
            _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=use_api_as_teacher)

        projection_f = partial(alfworld_projection_spark)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    
    elif "sciworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.sciworld import build_sciworld_envs, sciworld_projection_spark
        import json
        generalization_level = config.env.sciworld.generalization_level

        if generalization_level == 2:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L2_idx.json'
        elif generalization_level == 1:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L1_idx.json'
        elif generalization_level == 0:
            variation_path = 'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json'

        with open(variation_path, 'r') as f:
            variations_idx = json.load(f)

        simplifications_preset = config.env.sciworld.get('simplifications_preset', "easy")
        env_step_limit = config.env.sciworld.get('env_step_limit', 100)
        jar_path = config.env.sciworld.get('jar_path', None)

        _envs = build_sciworld_envs(
            seed=config.env.seed, 
            env_num=config.data.train_batch_size, 
            group_n=group_n, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx['train']
        )

        _val_envs = build_sciworld_envs(
            seed=config.env.seed + 1000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx['test'] if not use_api_as_teacher else variations_idx['train'],
            # variations_idx=variations_idx['test' if not config.env.use_api else "train"]
        )

        # Create projection function
        projection_f = partial(sciworld_projection_spark)

        # Create environment managers
        envs = SciWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = SciWorldEnvironmentManager(_val_envs, projection_f, config)

        # Give some time for environments to initialize
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1)

        return envs, val_envs
    
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection_spark
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        # val_is_train = True if config.env.use_api else False
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=use_api_as_teacher, env_kwargs=env_kwargs)

        projection_f = partial(webshop_projection_spark)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection_spark
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(sokoban_projection_spark)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection_spark
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)
        
        projection_f = partial(gym_projection_spark, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection_spark
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection_spark)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)