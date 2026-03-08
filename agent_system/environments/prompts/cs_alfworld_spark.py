######### SPARK TEMPLATE #########
ALFWORLD_TEMPLATE_NO_HIS_SPARK_CS= """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current task is: {task_description}
Your current observation is: {current_observation}

You have two reasoning modes:
<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
</explore>

<think>
Continuously track the current progress and history of reasoning and execution throughout the task.Typically used when task outcomes are as expected and no other mode of reasoning is required.
</think>

You need to choose one of the two reasoning modes mentioned above as your reasoning mode, based on its definition and applicable situation.

Now it's your turn to first reason and then take an action.
1.You should first reason about the current situation in one concise, brief sentence. This reasoning process MUST be enclosed within ONLY ONE of the above reasoning tags, like: <tag> your reasoning process </tag>.
2.Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags. 
"""

ALFWORLD_TEMPLATE_SPARK_CS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
To solve the problem, you have made the plan: [{planning}].

You have two reasoning modes:

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
</explore>

<think>
Continuously track the current progress and history of reasoning and execution throughout the task.Typically used when task outcomes are as expected and no other mode of reasoning is required.
</think>

You need to choose one of the two reasoning modes mentioned above as your reasoning mode, based on its definition and applicable situation.

Now it's your turn to first reason and then take an action.
1.You should first reason about the current situation in one concise, brief sentence. This reasoning process MUST be enclosed within ONLY ONE of the above reasoning tags, like: <tag> your reasoning process </tag>.
2.Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags. 
"""

######### TAGGING TEMPLATE #########
ALFWORLD_TAGGING_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment.  
I will provide you with a successful trajectory. You need to supplement the reasoning process.

**Reason using ONLY ONE tag pair and express your reasoning step by step:**

<explore>
When immediate next steps have a clear exploratory nature—such as when searching for an unknown object or information—use current observations to think outside the box and generate as many possible hypotheses, locations, items, or actions as possible.
Employ this approach when results are unexpected, information is lacking, or obstacles demand creative and innovative problem-solving.
</explore>

<think>  
Continuously track the current progress and history of reasoning and execution throughout the task.
Firstly recall the current subgoal based on the previously established overall plan, then consider the next action required to achieve this subgoal.
Typically used whe task outcomes are as expected and no other mode of reasoning is required.
</think>

You need to output a list in JSON format, with the same length as the trajectory. Each element should contain two key-value pairs, for example:  
```json
[{{"reason": "<explore>The book may be in the cabinet, shelf, so in the next steps I need to search these locations.</explore>", "action": "go to shelf 1"}},
{{"reason": "<think>Currently, my sub-goal is to obtain item A. I have already spotted A, and in order to accomplish this objective, I need to pick it up.</think>", "action": "pick up A"}}]
```
The "action" field must match the action in the trajectory, and the "reason" field should be a reasonable reasoning process inferred from the context of previous actions and the next few actions.

now the trajectory is as follows: {traj}
"""

