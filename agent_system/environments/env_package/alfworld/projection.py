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

import re
from typing import List, Tuple, Optional, Dict, Any

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
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
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids, [], valids



import re
from typing import List, Dict, Tuple, Optional

def alfworld_projection_spark(
    actions: List[str], 
    action_pools: List[List[str]]
) -> Tuple[List[str], List[int], Dict[str, List[Optional[str]]], List[bool]]:

    # --- 1. Define fixed reasoning tags ---
    REASONING_TAGS_PRIORITY: List[str] = ['explore', 'think']

    # --- 2. Compile all regular expressions dynamically and once ---
    ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
    CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')
    REASONING_PATTERNS: Dict[str, re.Pattern] = {
        tag: re.compile(f"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
        for tag in REASONING_TAGS_PRIORITY
    }

    def _extract_first_tag_content(pattern: re.Pattern, text: str) -> Optional[str]:
        """
        Helper function to search for the first match of a pattern and extract its content.
        """
        match = pattern.search(text)
        if not match:
            return None
        
        # 1. Check inner content (group 1)
        inner_content = match.group(1).strip()
        if inner_content:
            # 2. If content exists, return the full match (group 0) including tags
            return match.group(0).strip()
        else:
            # 3. If content is empty (e.g., "<think></think>"), 
            #    treat as invalid and return None
            return None
    

    actions_out = []
    valids = []
    action_available = []

    # <--- MODIFIED: Initialize reasoning_outputs as a dictionary of lists ---
    # Includes keys for each priority tag, the prioritized 'reasoning', and 'active_reasoning_tag'
    all_reasoning_keys = REASONING_TAGS_PRIORITY + ["reasoning"] + ["active_reasoning_tag"]
    reasoning_outputs: Dict[str, List[Optional[str]]] = {
        key: [] for key in all_reasoning_keys
    }

    for i, response in enumerate(actions):
        valid = True
        act_str = ""

        # --- 3. & 4. Dynamic content extraction and prioritization ---
        prioritized_reasoning = None
        active_reasoning_tag = None # Determines logic branching based on priority
        
        # Temporary storage for extracted values in the current iteration
        extracted_content_this_loop: Dict[str, Optional[str]] = {}

        for tag in REASONING_TAGS_PRIORITY:
            pattern = REASONING_PATTERNS[tag]
            content = _extract_first_tag_content(pattern, response)
            
            extracted_content_this_loop[tag] = content
            
            # Check priority (the first non-empty tag content found)
            if content and prioritized_reasoning is None:
                prioritized_reasoning = content
                active_reasoning_tag = tag
        
        # <--- MODIFIED: Append extracted values to respective lists ---
        for tag in REASONING_TAGS_PRIORITY:
            reasoning_outputs[tag].append(extracted_content_this_loop[tag])
        
        # Append prioritized results
        reasoning_outputs["reasoning"].append(prioritized_reasoning)
        reasoning_outputs["active_reasoning_tag"].append(active_reasoning_tag)

        # --- 5. Validation: Check for Chinese characters ---
        if CHINESE_RE.search(response):
            valid = False

        # --- 6. Validation: Action tags ---
        action_matches = list(ACTION_RE.finditer(response))
        min_action_pos = float('inf')

        if len(action_matches) != 1:
            valid = False
            action_available.append(False)
        else:
            action_match = action_matches[0]
            act_str = action_match.group(1).strip()
            min_action_pos = action_match.start()
            # Verify if the action exists in the predefined action pool for this step
            action_available.append(act_str in action_pools[i])

        # --- 7. Dynamic Validation: Reasoning tags ---
        all_reasoning_matches = []
        for pattern in REASONING_PATTERNS.values():
            all_reasoning_matches.extend(list(pattern.finditer(response)))

        # Rule: Exactly one reasoning tag should be present
        if len(all_reasoning_matches) != 1:
            valid = False

        found_skill_content = False
        for match in all_reasoning_matches:
            if match.group(1).strip():
                found_skill_content = True
            # Rule: Reasoning tags must appear BEFORE the action tag
            if match.start() > min_action_pos:
                valid = False

        if not found_skill_content:
            valid = False

        # # --- 8. Modify act_str based on tag ---
        # if active_reasoning_tag == 'explore':
        #     act_str = "EXPLORE"

        # --- 9. Collect results ---
        actions_out.append(act_str)
        valids.append(int(valid)) 

    # <--- MODIFIED: Returns 'reasoning_outputs' as a dictionary of lists
    return actions_out, valids, reasoning_outputs, action_available