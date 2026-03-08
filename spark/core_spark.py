import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
import re

def compute_spark_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.array, # ['uid']
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    reasoning_modes: np.array = None,
    traj_id: np.array = None
):
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, epsilon, remove_std)
    step_advantages = step_norm_reward_by_tag(
        step_rewards=token_level_rewards.sum(dim=-1),
        response_mask=response_mask,
        index=index,
        reasoning_modes=reasoning_modes,
        epsilon=epsilon,
        remove_std=remove_std,
        traj_id=traj_id
    )
    ## exclude think

    scores = episode_advantages + 0*step_advantages
    return scores, scores

def episode_norm_reward(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.array,
    epsilon: float = 1e-6,
    remove_std: bool = True,
):
    # 会因为steps的数量而引入一定偏差
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    return episode_advantages

# not used currently
def process_trajectory_spark_rewards(
    trajectory_list,
    config,
    episode_rewards
):
    # planning_reward_values = []
    # exploration_reward_values = []
    # reflection_reward_values = []
    def tag_count_of_traj(trajectory: list):
        count_dict = {}
        count_dict['<planning>'] = 0
        count_dict['<explore>'] = 0
        count_dict['<think>'] = 0
        for step_idx, step_data in enumerate(trajectory):
            full_output = step_data['full_output']
            if re.search(r'<planning>.*?</planning>', full_output, re.DOTALL):
                count_dict['<planning>'] += 1
            if re.search(r'<explore>.*?</explore>', full_output, re.DOTALL):
                count_dict['<explore>'] += 1
            if re.search(r'<think>.*?</think>', full_output, re.DOTALL):
                count_dict['<think>'] += 1 
        return count_dict
    
    for traj_idx, trajectory in enumerate(trajectory_list):
        count_dict = tag_count_of_traj(trajectory)
        ## if this episode success
        won = episode_rewards[traj_idx] > 0
        steps = len(trajectory)
        ## explore比例高于think
        if count_dict['<explore>'] > count_dict['<think>']:
            if won:
                episode_rewards[traj_idx] = 5
            else:
                episode_rewards[traj_idx] = -0.1
        ## 恰好在max_steps成功  
        if won and steps == config.env.max_steps:
            episode_rewards[traj_idx] = 5
            
        ## No planning or no explore
        if count_dict['<planning>'] + count_dict['<explore>']< 2: 
            episode_rewards[traj_idx] -= 0.1

        # planning_cnt = 0
        # explore_cnt = 0
        # think_cnt = 0
        # trajectory_out = ''

        # for step_idx, step_data in enumerate(trajectory):
        #     full_output = step_data['full_output']
        #     trajectory_out = trajectory_out + full_output
        #     if re.search(r'<planning>.*?</planning>', full_output, re.DOTALL):
        #         planning_cnt += 1
        #     if re.search(r'<explore>.*?</explore>', full_output, re.DOTALL):
        #         explore_cnt += 1
        #     if re.search(r'<think>.*?</think>', full_output, re.DOTALL):
        #         think_cnt += 1
        # if planning_cnt > 0 and explore_cnt > 0 and think_cnt > 0:
        #     episode_rewards[traj_idx] += config.algorithm.spark.explore_reward

            # ### 惩罚回复中出现了不止一种的reasoning tag
            # if count_all_tags(full_output) > 1:
            #     episode_rewards[traj_idx] += config.algorithm.spark.reason_format_reward
            
            ### 奖励explore
            # if re.search(r'<explore>.*?</explore>', full_output, re.DOTALL):
            #     episode_rewards[traj_idx] += config.algorithm.spark.explore_reward
            #     break

# not used currently
def step_norm_reward_by_tag(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.array,
    reasoning_modes: np.array,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    traj_id: np.array = None, 
):
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()
    group_keys = []
    for i in range(len(index)):
        group_key = (index[i], reasoning_modes[i])
        group_keys.append(group_key)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            group_key = group_keys[i]
            id2score[group_key].append(scores[i])
        for group_key in id2score:
            if len(id2score[group_key]) == 1:
                id2mean[group_key] = torch.tensor(0.0)
                id2std[group_key] = torch.tensor(1.0)
            elif len(id2score[group_key]) > 1:
                id2mean[group_key] = torch.mean(torch.tensor(id2score[group_key]))
                id2std[group_key] = torch.std(torch.tensor(id2score[group_key]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[group_key]): {len(id2score[group_key])}")
                raise ValueError(f"no score in group: {group_key}")
        for i in range(bsz):
            group_key = group_keys[i]
            if remove_std:
                scores[i] = scores[i] - id2mean[group_key]
            else: 
                scores[i] = (scores[i] - id2mean[group_key]) / (id2std[group_key] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    return step_advantages 


