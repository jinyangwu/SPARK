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


def sokoban_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    Sokoban action mappings:
    - 0: Still (Invalid Action)
    - 1: Up
    - 2: Down
    - 3: Left
    - 4: Right
    """

    action_pools = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "still": 0,
    }

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = 0  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()

            for act in action_pools.keys():
                if act in extracted_action:
                    actions[i] = action_pools[act]
                    # if found legal action, set valids to 1
                    valids[i] = 1
                    break

            # If no valid action found, randomly select from pool
            if valids[i] == 0:
                actions[i] = 0

        except:
            # randomly choose an action from the action list if illegal
            actions[i] = 0

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

    return actions, valids

import re
from typing import List
def sokoban_projection_spark(actions: List[str]):
    """
    Project actions for Sokoban task with strict validation logic aligned with Webshop logic.
    
    Sokoban action mappings:
    - 0: Still (Invalid Action)
    - 1: Up
    - 2: Down
    - 3: Left
    - 4: Right
    """
    
    # Define Sokoban-specific action mappings
    action_map = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "still": 0,
    }

    skill_tags = [
        r"<planning>.*?</planning>",
        r"<explore>.*?</explore>",
        r"<think>.*?</think>"
    ]

    actions_out = []        # Stores converted integer actions (0-4)
    valids = []             # Stores flag indicating if the sample is valid (0 or 1)
    plannings = []          # Stores extracted planning content for analysis
    action_available = [False] * len(actions) # Flags whether the action exists in the valid mapping

    for i, output in enumerate(actions):
        valid = 1
        act_int = 0  # Default to 0 (Still/Invalid)

        # 1. Extract Planning content (useful for potential reward shaping or post-analysis)
        planning_content = None
        planning_match = re.search(r"<planning>([\s\S]*?)</planning>", output, re.IGNORECASE)
        if planning_match:
            planning_inner = planning_match.group(1).strip()
            planning_content = planning_inner if planning_inner else None
        plannings.append(planning_content)

        # 2. Chinese character check (strict filtering for English-only environments)
        if re.search(r'[\u4e00-\u9fff]', output):
            valid = 0

        # 3. Validation: <action> tags must occur exactly once and match the mapping table
        matches = re.findall(r"<action>([\s\S]*?)</action>", output)
        if len(matches) != 1:
            valid = 0
        else:
            act_candidate = matches[0].strip().lower()
            if act_candidate in action_map:
                act_int = action_map[act_candidate]
                action_available[i] = True
            else:
                # Treat as invalid if the action is not in the allowed keys (e.g., "<action>jump</action>")
                valid = 0 

        # 4. Validation: Skill tags (Think/Planning/Explore)
        # Logic: Exactly one non-empty skill tag must exist and be placed BEFORE the <action> tag
        found_skill = False
        min_action_pos = output.lower().find("<action>")
        if min_action_pos == -1: 
            min_action_pos = float('inf') # Prevent logic errors when no action tag is found

        skill_count = 0
        for tag in skill_tags:
            tag_matches = list(re.finditer(tag, output, re.IGNORECASE | re.DOTALL))
            skill_count += len(tag_matches)
            for tag_match in tag_matches:
                if tag_match:
                    # Strip XML tags to check if internal content is non-empty
                    inner = re.sub(r"<.*?>", "", tag_match.group(0)).strip()
                    if inner:
                        found_skill = True
                        # Requirement: Skill tag must appear before the <action> tag
                        if output.find(tag_match.group(0)) > min_action_pos:
                            valid = 0

        # Strict requirement: Exactly one skill tag must be present
        if skill_count != 1:  
            valid = 0
        # Requirement: Skill tag content cannot be empty
        if not found_skill:
            valid = 0

        actions_out.append(act_int)
        valids.append(valid)

    return actions_out, valids, plannings, action_available