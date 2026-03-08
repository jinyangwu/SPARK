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

from typing import List
import re

def webshop_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    """

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
                actions[i] = actions[i][-20:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            # randomly choose an action from the action list if illegal
            actions[i] = actions[i][-20:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids


def webshop_projection_spark(actions: List[str], action_pools: List[List[str]]):
    skill_tags = [
        r"<planning>.*?</planning>",
        # r"<reflection>.*?</reflection>",
        r"<explore>.*?</explore>",
        r"<think>.*?</think>"
    ]

    actions_out = []
    valids = []
    plannings = []
    action_available = [False] * len(actions)
    for i, output in enumerate(actions):
        valid = 1
        act_str = ""

        planning_content = None
        planning_match = re.search(r"<planning>([\s\S]*?)</planning>", output, re.IGNORECASE)
        if planning_match:
            planning_inner = planning_match.group(1).strip()
            planning_content = planning_inner if planning_inner else None
        plannings.append(planning_content)

        # Check for Chinese
        if re.search(r'[\u4e00-\u9fff]', output):
            valid = 0

        # Check ONLY ONE <action>...</action>
        matches = re.findall(r"<action>([\s\S]*?)</action>", output)
        if len(matches) != 1:
            valid = 0
        else:
            act_candidate = matches[0].strip()
            act_str = act_candidate
            if act_candidate in action_pools[i]:
                action_available[i] = True

        # Check ONLY ONE skill tag, appears before <action> and is non-empty
        found_skill = False
        min_action_pos = output.lower().find("<action>")
        skill_positions = []
        skill_count = 0
        for tag in skill_tags:
            tag_matchs = list(re.finditer(tag, output, re.IGNORECASE | re.DOTALL))
            skill_count += len(tag_matchs)
            for tag_match in tag_matchs:
                if tag_match:
                    # Remove the xml tags and check if not empty
                    inner = re.sub(r"<.*?>", "", tag_match.group(0)).strip()
                    if inner:
                        found_skill = True
                        skill_positions.append(output.find(tag_match.group(0)))
                        # skill tag must appear before <action>
                        if output.find(tag_match.group(0)) > min_action_pos:
                            valid = 0

        if skill_count != 1:  
            valid = 0
        if not found_skill:
            valid = 0

        actions_out.append(act_str)
        valids.append(valid)

    return actions_out, valids, plannings, action_available