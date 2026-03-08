import numpy as np
import uuid
from typing import Dict, List, Tuple, Optional, Callable
from copy import deepcopy
import random
from verl import DataProto
import math, re

def extract_batch_with_mask(expanded_gen_batch: DataProto, active_indices: np.ndarray) -> DataProto:
    """Extract active parts from the expanded gen_batch using a mask."""
    active_batch = expanded_gen_batch[active_indices]
    return active_batch

def extract_obs_with_mask(obs: Dict, active_indices: np.ndarray) -> Dict:
    """Extract active parts from observations using a mask."""
    active_obs = {}
    for key, value in obs.items():
        if isinstance(value, list):
            active_obs[key] = [value[i] for i in active_indices if i < len(value)]
        elif isinstance(value, np.ndarray):
            valid_indices = active_indices[active_indices < len(value)]
            active_obs[key] = value[valid_indices] if len(valid_indices) > 0 else value
        else:
            active_obs[key] = value
    return active_obs

def pad_and_rebuild_all_trajectory_data(
    total_batch_list, episode_rewards, episode_lengths, traj_uid, total_infos,
    sample_to_rollouts_map, max_rollouts_per_sample, num_samples, used_indices
):
    """
    Uniformly handles trajectory padding and info reconstruction.
    Ensures each sample has the required number of rollouts for batch processing.
    """
    final_total_batch_list = []
    final_episode_rewards = []
    final_episode_lengths = []
    final_traj_uid = []
    final_total_infos = []
    
    for sample_idx in range(num_samples):
        sample_rollout_indices = sample_to_rollouts_map.get(sample_idx, [])
        valid_sample_indices = [i for i, original_idx in enumerate(used_indices) 
                               if original_idx in sample_rollout_indices]
        current_count = len(valid_sample_indices)
        
        # Add existing valid trajectories
        for idx in valid_sample_indices:
            final_total_batch_list.append(total_batch_list[idx])
            final_episode_rewards.append(episode_rewards[idx])
            final_episode_lengths.append(episode_lengths[idx])
            final_traj_uid.append(traj_uid[idx])
            final_total_infos.append(total_infos[idx])
        
        # Supplement missing trajectories via random resampling
        if current_count < max_rollouts_per_sample and current_count > 0:
            needed = max_rollouts_per_sample - current_count
            for _ in range(needed):
                source_idx = random.choice(valid_sample_indices)
                final_total_batch_list.append(deepcopy(total_batch_list[source_idx]))
                final_episode_rewards.append(episode_rewards[source_idx])
                final_episode_lengths.append(episode_lengths[source_idx])

                # Generate new unique ID for the duplicated trajectory
                final_traj_uid.append(str(uuid.uuid4()))
                for i in range(len(final_total_batch_list[-1])):
                   final_total_batch_list[-1][i]['traj_uid'] = final_traj_uid[-1]
                
                final_total_infos.append(deepcopy(total_infos[source_idx])) 
        elif current_count == 0:
            # Placeholder for empty samples
            for _ in range(max_rollouts_per_sample):
                final_total_batch_list.append([])
                final_episode_rewards.append(0.0)
                final_episode_lengths.append(0)
                final_traj_uid.append(str(uuid.uuid4()))
                final_total_infos.append([])
    
    return (final_total_batch_list, 
            np.array(final_episode_rewards, dtype=np.float32), 
            np.array(final_episode_lengths, dtype=np.int32),
            np.array(final_traj_uid, dtype=object),
            final_total_infos)

def update_obs_with_mask(full_obs, next_obs, active_indices, all_branches: dict):
    """Update the output from masked environments back into the full observation structure."""
    updated_obs = deepcopy(full_obs)
    assert len(next_obs['text']) == len(active_indices)
    
    # Update indices that were actively stepped
    for i, active_idx in enumerate(active_indices):
        if 'text' in updated_obs:
            updated_obs['text'][active_idx] = next_obs['text'][i]
        if 'image' in updated_obs and next_obs['image'] is not None:
            updated_obs['image'][active_idx] = next_obs['image'][i]  
        if 'anchor' in updated_obs:
            updated_obs['anchor'][active_idx] = next_obs['anchor'][i]
            
    # Synchronize branched children with their parent states
    for parent_idx, childs_idx in all_branches.items():
        for child_idx in childs_idx:
            if 'text' in updated_obs:
                updated_obs['text'][child_idx] = updated_obs['text'][parent_idx]
            if 'image' in updated_obs and next_obs['image'] is not None:
                updated_obs['image'][child_idx] = updated_obs['image'][parent_idx]  
            if 'anchor' in updated_obs:
                updated_obs['anchor'][child_idx] = updated_obs['anchor'][parent_idx]

    return updated_obs

def update_infos_with_mask(full_infos, next_infos, active_indices, all_branches):
    """Update masked environment infos back into the full infos list."""
    updated_infos = deepcopy(full_infos)
    assert len(next_infos) == len(active_indices)
    
    for i, active_idx in enumerate(active_indices):
        updated_infos[active_idx] = next_infos[i]
    
    for parent_idx, childs_idx in all_branches.items():
        for child_idx in childs_idx:
             updated_infos[child_idx] = updated_infos[parent_idx]

    return updated_infos

def calculate_entropy_from_logprobs(logprobs_data, vocab_size: int) -> float:
    """
    Calculate normalized entropy from logprobs data.
    
    Args:
        logprobs_data: Logprobs from model output (vLLM format, Dict, or List)
        vocab_size: Size of vocabulary for normalization
    
    Returns:
        Normalized entropy value between 0 and 1
    """
    if not logprobs_data:
        return 0.0
    
    # Extract logprobs based on input format
    if hasattr(logprobs_data, 'values'):
        # vLLM format
        token_list = list(logprobs_data.values())
        logprobs = [token.logprob for token in token_list]
    elif isinstance(logprobs_data, dict):
        logprobs = list(logprobs_data.values())
    elif isinstance(logprobs_data, list):
        logprobs = logprobs_data
    else:
        return 0.0
    
    # Shannon Entropy formula: H = -sum(p * log(p))
    p_list = [math.exp(l) for l in logprobs]
    entropy = -sum(p * l for p, l in zip(p_list, logprobs))
    
    # Normalize by log(vocab_size)
    entropy_norm_factor = math.log(vocab_size)
    return entropy / entropy_norm_factor if entropy_norm_factor > 0 else 0.0

def adaptive_entropy_branching_condition(
    step: int,
    initial_entropy_dict: Dict[int, float],
    current_entropy_dict: Dict[int, float],
    active_indices: np.ndarray,
    branch_probability: float = 0.5,
    entropy_weight: float = 0.5,
    min_step_for_branching: int = 0
) -> np.ndarray:
    """
    Adaptive branching logic based on entropy change.
    
    Args:
        step: Current rollout step
        initial_entropy_dict: Starting entropy for each rollout
        current_entropy_dict: Latest calculated entropy
        active_indices: Indices of rollouts currently being processed
        branch_probability: Base probability for a branch to occur
        entropy_weight: Sensitivity factor for entropy delta
        min_step_for_branching: Minimum steps required before branching starts
    """
    should_branch = []
    
    for i, rollout_idx in enumerate(active_indices):
        try:
            entropy_now = current_entropy_dict.get(rollout_idx, 0.0)
            entropy_init = initial_entropy_dict.get(rollout_idx, 0.0)
            entropy_delta = entropy_now - entropy_init
            
            # Higher entropy increases the probability of branching (more uncertainty)
            prob = branch_probability + entropy_weight * entropy_delta
            prob = max(0.0, min(1.0, prob))  # Clamp to [0, 1]
            
            branch_decision = random.random() < prob
            should_branch.append(branch_decision)
            
        except Exception as e:
            print(f"Error in adaptive branching for rollout {rollout_idx}: {e}")
            should_branch.append(False)
    
    return np.array(should_branch, dtype=bool)

def _calc_entropy(logprobs: List[float]) -> float:
    """Calculate raw Shannon entropy from a list of log probabilities."""
    if not logprobs:
        return 0.0
    p_list = [math.exp(l) for l in logprobs]
    entropy = -sum(p * l for p, l in zip(p_list, logprobs))
    return entropy

def entropy_variation_monitoring(
    entropys,  
    active_indices: np.ndarray,
    tokenizer,
    initial_entropy_dict: Dict[int, float]
) -> Dict[int, float]:
    """
    Monitors variations in entropy. Supports tensor format logprobs.
    
    Args:
        entropys: List of entropy values corresponding to active indices
        active_indices: Array of currently active rollout indices
        tokenizer: Tokenizer object
        initial_entropy_dict: Dictionary storing baseline entropy (modified in-place)
        
    Returns:
        Dict[int, float]: Dictionary mapping rollout index to current entropy
    """
    current_entropy_dict = {}
    
    for i, out_idx in enumerate(active_indices):
        current_entropy_dict[out_idx] = entropys[i]
        # Set the baseline if this is the first time observing this index
        if out_idx not in initial_entropy_dict:
            initial_entropy_dict[out_idx] = entropys[i]
    
    return current_entropy_dict

def explore_branching_condition(
    step: int,
    text_actions: list[str],
) -> np.ndarray:
    """Trigger branching if the text action contains a specific <explore> tag."""
    should_branch = []
    for text_action in text_actions:
        # Regex to match the complete <explore>...</explore> tag structure
        if re.search(r"<explore>.*?</explore>", text_action, re.DOTALL):
            should_branch.append(True)
        else:
            should_branch.append(False)
        
    return np.array(should_branch, dtype=bool)

def fixed_branching_condition(
    step: int,
    text_actions: list[str],
    num_branches: int = 2,  # Fixed number of branches to generate per step
) -> np.ndarray:
    """Randomly selects a fixed number of actions to trigger branching."""
    num_actions = len(text_actions)
    num_branches = min(num_branches, num_actions)

    # Randomly select which indices will branch
    selected_indices = random.sample(range(num_actions), num_branches)
    should_branch = np.zeros(num_actions, dtype=bool)

    for idx in selected_indices:
        should_branch[idx] = True

    return should_branch

if __name__ == "__main__":
    # Test logic
    text_actions = ["<explore></explore>", "test", "<explore>"]
    should_branch = explore_branching_condition(0, text_actions)
    print(should_branch)