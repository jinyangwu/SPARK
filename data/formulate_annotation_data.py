import json
import re
import jsonlines,os
from agent_system.environments.prompts.cs_alfworld_spark import ALFWORLD_TEMPLATE_NO_HIS_SPARK_CS, ALFWORLD_TEMPLATE_SPARK_CS
from agent_system.environments.prompts.cs_sciworld_spark import SCIWORLD_TEMPLATE_NO_HIS_SPARK_CS, SCIWORLD_TEMPLATE_SPARK_CS
from agent_system.environments.prompts.cs_webshop_spark import WEBSHOP_TEMPLATE_NO_HIS_SPARK_CS, WEBSHOP_TEMPLATE_SPARK_CS

ENV_NAME = "alfworld"  # Options: alfworld, sciworld, webshop
GENERALIZATION_LEVEL = "L0"  # Options: L0, L1, L2
DIR_BASE = f"data/{ENV_NAME}_cs_{GENERALIZATION_LEVEL}"
TEACHER_RES_SAVE_PATH = os.path.join(DIR_BASE, f"{ENV_NAME}_annotation.jsonl")
SAVE_PATH =  os.path.join(DIR_BASE, f"{ENV_NAME}_cold-start_annotation.json")
HISTORY_LEN = 5
os.makedirs(DIR_BASE, exist_ok=True)


if ENV_NAME == "alfworld":
    TEMPLATE_NO_HIS_SPARK_CS = ALFWORLD_TEMPLATE_NO_HIS_SPARK_CS
    TEMPLATE_SPARK_CS = ALFWORLD_TEMPLATE_SPARK_CS
elif ENV_NAME == "sciworld":
    TEMPLATE_NO_HIS_SPARK_CS = SCIWORLD_TEMPLATE_NO_HIS_SPARK_CS
    TEMPLATE_SPARK_CS = SCIWORLD_TEMPLATE_SPARK_CS
elif ENV_NAME == "webshop":
    TEMPLATE_NO_HIS_SPARK_CS = WEBSHOP_TEMPLATE_NO_HIS_SPARK_CS
    TEMPLATE_SPARK_CS = WEBSHOP_TEMPLATE_SPARK_CS


with jsonlines.open(TEACHER_RES_SAVE_PATH, 'r') as reader:
    meta_trajs = list(reader)

sft_data = []
for meta_traj in meta_trajs:
    task = meta_traj['task']
    res = meta_traj['traj']
    step_level_data = []
    latest_planning = "No plan."
    for i, item in enumerate(res):
        if i == 0:
            prompt = TEMPLATE_NO_HIS_SPARK_CS.format(
                task_description=task,
                current_observation=item["obs"],
            )
        else:
            history_think_length = min(HISTORY_LEN, i)
            lines = []
            pure_actions = []
            for j in range(i - history_think_length, i):
                step_num = j + 1
                obs = res[j]['obs']
                act = res[j]['action']
                lines.append(f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']")
                pure_actions.append(act)

            action_history = "\n".join(lines)
            prompt = TEMPLATE_SPARK_CS.format(
                task_description=task,
                step_count=i,
                history_length=history_think_length,
                action_history=action_history,
                pure_actions=pure_actions,
                current_step=i + 1,
                current_observation=item["obs"],
                planning=latest_planning
            )

        # update current planning
        if '<planning>' in item["reason"]:
            current_planning = re.search(r'<planning>(.*?)</planning>', item["reason"], re.DOTALL)
            if current_planning:
                latest_planning = current_planning.group(1).strip()
            else:
                pass

        response = f"{item['reason']}\n<action>{item['action']}</action>\n"

        step_level_data.append({
            "step": i + 1,
            "obs": item["obs"],
            "prompt": prompt,
            "response": response,
        })

    item_data = {
        "task": task,
        "done": "True",
        "data": step_level_data
    }
    sft_data.append(item_data)


with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=4)