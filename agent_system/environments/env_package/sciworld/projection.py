from typing import List
import re

def sciworld_projection(actions: List[str], available_actions=None, meta_think=False):
    valids = [0] * len(actions)
    action_available = [False] * len(actions)
    processed_actions = []

    for i in range(len(actions)):
        original_str = actions[i]
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = original_str.find(start_tag)
        end_idx = original_str.find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                processed_actions.append(original_str[-20:])
                continue
            extracted_action = original_str[start_idx + len(start_tag):end_idx].strip()
            processed_actions.append(extracted_action)
            valids[i] = 1
            env_available_actions = available_actions[i]
            if extracted_action in env_available_actions:
                action_available[i] = True
        except:
            processed_actions.append(original_str[-20:])
        if meta_think:
            if ("<planning>" not in original_str or "</planning>" not in original_str) and \
               ("<explore>" not in original_str or "</explore>" not in original_str) and \
               ("<reflection>" not in original_str or "</reflection>" not in original_str) and \
               ("<monitor>" not in original_str or "</monitor>" not in original_str):
                valids[i] = 0
        else:
            think_start_idx = original_str.find("<think>")
            think_end_idx = original_str.find("</think>")
            if think_start_idx == -1 or think_end_idx == -1:
                valids[i] = 0
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return processed_actions, valids, action_available


import re
from typing import List, Tuple, Optional

def sciworld_projection_spark(actions: List[str], action_pools: List[List[str]]):
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