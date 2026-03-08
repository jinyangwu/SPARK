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

import torch
import random
from typing import List
import re

def gym_projection(text_actions: List[str], env_name):
    output_indices = []
    valids = []
    if env_name == 'gym_cards/NumberLine-v0':
        action_list = ["-", "+"]
    elif env_name == 'gym_cards/Blackjack-v0':
        action_list = ["stand", "hit"]
    elif env_name == 'gym_cards/EZPoints-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    elif env_name == 'gym_cards/Points24-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
    else:
        raise NotImplementedError("Action list not implemented for this env!")
    for string in text_actions:
        if not isinstance(string, str):
            # directly output a random action if the string is not a string
            output_indices.append(-1)
            valids.append(0)
            continue
        string = string.lower()
        action_index = string.find('"action":')
        # Extract everything after "action":
        string = string[action_index:]
        contained_actions = []
        # For the 'gym_cards/Points24-v0' environment, handle '10' separately
        if 'points' in env_name.lower() and '10' in string:
            contained_actions.append('10')
            string = string.replace('10', '')  # Remove '10' to prevent it from being counted as '1'
        # Find all actions that are contained in the string
        for action in action_list:
            if action in string:
                contained_actions.append(action)
        # Remove duplicates by converting to a set and back to a list
        contained_actions = list(set(contained_actions))
        if len(contained_actions) == 1 and contained_actions[0] in action_list:
            # Only one keyword from action_list is in the string
            output_indices.append(action_list.index(contained_actions[0]))
            valids.append(1)
        else:
            # The string contains none or multiple keywords, randomly select an index from action_list
            output_indices.append(-1)
            valids.append(0)
    return output_indices, valids

def gym_projection_spark(text_actions: List[str], env_name):
    output_indices = []
    valids = []
    if env_name == 'gym_cards/NumberLine-v0':
        action_list = ["-", "+"]
    elif env_name == 'gym_cards/Blackjack-v0':
        action_list = ["stand", "hit"]
    elif env_name == 'gym_cards/EZPoints-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    elif env_name == 'gym_cards/Points24-v0':
        action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
    else:
        raise NotImplementedError("Action list not implemented for this env!")
    for output in text_actions:
        valid_action = 1
        act_candidate = None

        ## 匹配<action></action>标签内的内容, 确保只有一个合法动作被提取
        matches = re.findall(r"<action>([\s\S]*?)</action>", output)
        if len(matches) != 1:
            valid_action = 0
        else:
            act_candidate = matches[0].strip().lower()
            if act_candidate not in action_list:
                valid_action = 0

        if valid_action == 1 and act_candidate in action_list:
            output_indices.append(action_list.index(act_candidate))
        else:
            output_indices.append(-1)

        valid_thought = 1
        if re.search(r'[\u4e00-\u9fff]', output):
            valid_thought = 0

        ## 两个都没找到 reasoning tags 则标记为无效
        if re.search(r"<explore>.*?</explore>", output, re.DOTALL) == None and re.search(r"<think>.*?</think>", output, re.DOTALL) == None:
            valid_thought = 0  # mark as valid if reasoning tags are found
  
        valid = valid_action * valid_thought
        valids.append(valid)

    return output_indices, valids