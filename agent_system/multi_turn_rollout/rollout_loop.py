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
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import Dict, List, Tuple, Optional, Callable
import pdb
from agent_system.multi_turn_rollout.branch_utils import *
from spark import core_gtpo
from agent_system.multi_turn_rollout.api import batch_infer
from collections import Counter
def aggregate_node_data(source_list):
    """
    Aggregates data entries based on their unique node_uid.
    Combines rewards and lengths from multiple occurrences of the same node.
    """
    # Use a dictionary as an "accumulator" with node_uid as the key.
    # This allows for O(1) lookups to find and update existing entries quickly.
    mapper = {}

    for item in source_list:
        uid = item['node_uid']
        
        # Extract data from the current item (default to empty list if not present)
        curr_rewards = item.get('episode_rewards', [])
        curr_lengths = item.get('episode_lengths', [])

        # --- Helper Function: Standardize data into a list ---
        # Whether the input is a single value (e.g., 10) or a list (e.g., [10]),
        # we convert it to a list to maintain consistency for merging.
        def to_list(val):
            return val if isinstance(val, list) else [val]

        if uid not in mapper:
            # 1. First time encountering this uid: Initialize the entry
            new_entry = item.copy()
            
            # Initialize the repetition counter
            new_entry['node_repeated_times'] = 1
            
            # Ensure these fields are initialized as new lists 
            # (Prevents reference issues and prepares them for future appending)
            new_entry['episode_rewards'] = list(to_list(curr_rewards))
            new_entry['episode_lengths'] = list(to_list(curr_lengths))
            
            mapper[uid] = new_entry
        else:
            # 2. Duplicate uid encountered: Aggregate the data
            existing_entry = mapper[uid]
            
            # Increment the repetition counter
            existing_entry['node_repeated_times'] += 1
            
            # Merge list data (using extend to flatten lists)
            existing_entry['episode_rewards'].extend(to_list(curr_rewards))
            existing_entry['episode_lengths'].extend(to_list(curr_lengths))

    # Convert the dictionary values back into a list of dictionaries
    return list(mapper.values())

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.branch_info = {}
        self.all_step_branches = []

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        try:
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                                tokenizer=self.tokenizer,
                                                                                max_length=self.config.data.max_prompt_length,
                                                                                pad_token_id=self.tokenizer.pad_token_id,
                                                                                left_pad=True,
                                                                                truncation=self.config.data.truncation,)
        except Exception as e:
            # If any error occurs (NameError, TypeError, ValueError, etc.), it will be caught here.
            print("❌ An error occurred during tokenization!")
            print(f"Error details: {e}")
            print("-" * 30)
            # This is where you fulfill your request to print the problematic variable
            print("Content of prompt_with_chat_template:")
            print(prompt_with_chat_template)
            with open('/raid1/HOME/szhang/jywu/code/Planning/GTPO/a.txt','w') as f:
                print(f.write(prompt_with_chat_template))
            raise
            

        if is_multi_modal:

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            is_train
            ) :
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        # 这里的batch_size是所有rollout的数量，正常为16*8=128
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            sample_intervel = self.config.env.rollout.n if is_train else 1
            sample_id = int(bs / sample_intervel)
            traj_id = f"{sample_id}_{bs%sample_intervel}" # like, 0_0, 0_1 , ... , 0_7; 1_0, 1_1 , ... , 1_7.....
            for i, data in enumerate(total_batch_list[bs]):
                # 每个rollout中所有的steps
                step_id = f"{traj_id}_{i+1}"
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = [episode_rewards[bs]]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = [episode_lengths[bs]]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    # sample_id and step_id 
                    data['step_id'] = step_id
                    data['node_repeated_times'] = 1
                    effective_batch.append(data)
        

        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )

        if is_train and  self.config.env.use_branch:
            effective_batch_unique = aggregate_node_data(effective_batch)
            gen_batch_output_unique = DataProto.from_single_dict(
                data=collate_fn(effective_batch_unique)
            )
            print("****GATHER DATASIZE", len(effective_batch), len(effective_batch_unique))
        else:
            gen_batch_output_unique = None

        ## 输出的data bsz为 steps*rollouts*samples
        return gen_batch_output, gen_batch_output_unique

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """
        # gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        # Initial observations from the environment
        # if self.config.env.env_name == 'search':
        #     obs, infos = envs.reset(gen_batch.non_tensor_batch['env_kwargs'])
        # else:
        obs, infos = envs.reset()

        # Initialize trajectory collection
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        if len(gen_batch.batch) != lenght_obs and self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        batch_size = len(gen_batch.batch['input_ids']) #这里已经变为batch_size*rollouts了
        batch_output = None
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object) # 为每个sample分配一个uid
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object) # 为每个rollout分配一个uid

        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            
            # import pdb
            # pdb.set_trace()
            # batch.non_tensor_batch['raw_prompt']
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

            multi_modal_data = None
            if "multi_modal_data" in batch.non_tensor_batch:
                multi_modal_data = batch.non_tensor_batch["multi_modal_data"]
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = gen_batch.meta_info

            batch_output = actor_rollout_wg.generate_sequences(batch_input)
           
            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)

            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            # Create reward tensor, only assign rewards for active environments
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            batch.non_tensor_batch['full_output'] = deepcopy(text_actions)
            if multi_modal_data is not None:
                batch.non_tensor_batch['multi_modal_data'] = multi_modal_data
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        ### add wins
        final_wons = [reward > 0 for reward in episode_rewards]
        for traj_idx, traj in enumerate(total_batch_list):
            for step_idx, step_data in enumerate(traj):
                step_data['won'] = int(final_wons[traj_idx])

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid
    
    def api_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """
        # Initial observations from the environment
        obs, infos = envs.reset()

        # Initialize trajectory collection
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        if len(gen_batch.batch) != lenght_obs and self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        batch_size = len(gen_batch.batch['input_ids']) #这里已经变为batch_size*rollouts了
        batch_output = None
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object) # 为每个sample分配一个uid
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object) # 为每个rollout分配一个uid

        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            
            # import pdb
            # pdb.set_trace()
            # batch.non_tensor_batch['raw_prompt']
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = gen_batch.meta_info

            if "multi_modal_data" in batch_input.non_tensor_batch:
                api_inputs = []
                multi_modal_data = batch_input.non_tensor_batch['multi_modal_data']
                raw_prompts = batch_input.non_tensor_batch['raw_prompt']
                # import pdb
                # pdb.set_trace()
                for p, img in zip(raw_prompts, multi_modal_data):
                    api_inputs.append({
                        "image": img['image'][0],  # 取出图像数据
                        "text": p[0]['content']    # 取出文本数据
                    })
            else:
                api_inputs = [p[0]['content'] for p in batch_input.non_tensor_batch['raw_prompt']]            
            print("STEP:", _step)
            provider = self.config.env.api.provider
            model_name = self.config.env.api.model_name
            max_workers = self.config.env.api.max_workers
            responses = batch_infer(api_inputs, active_masks, provider, model_name, max_workers)

            dummy_tensors = {"input_ids": torch.zeros(len(responses)), "responses": torch.zeros(len(responses))}
            batch_output = DataProto.from_dict(tensors=dummy_tensors,
                                            non_tensors={'full_output': responses, 'raw_prompt': batch_input.non_tensor_batch['raw_prompt']})
           

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            if "multi_modal_data" in batch_input.non_tensor_batch:
                batch.non_tensor_batch['multi_modal_data'] = batch_input.non_tensor_batch['multi_modal_data']


            # pdb.set_trace()
            batch = batch.union(batch_output)

            text_actions = responses
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            # Create reward tensor, only assign rewards for active environments
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        ### add wins
        final_wons = [reward > 0 for reward in episode_rewards]
        for traj_idx, traj in enumerate(total_batch_list):
            for step_idx, step_data in enumerate(traj):
                step_data['won'] = int(final_wons[traj_idx])

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid


    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )

        elif self.config.env.use_branch and is_train:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.branching_multi_turn_loop_with_masks(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                branch_count=self.config.env.branch_count
            )
        elif self.config.env.use_api:
            # api loop
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.api_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        

        # Create trajectory data
        gen_batch_output = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            is_train=is_train
        )
        
        return gen_batch_output
        
def branching_multi_turn_loop_with_masks(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            branch_count: int = 2,
            ) -> Tuple[List, np.ndarray, np.ndarray, Dict, np.ndarray, Dict]:
        """
        Executes a multi-turn rollout loop with support for dynamic branching and masking.
        Branches are managed to ensure they stay within the pre-allocated environment limits.
        """

        # Retrieve rollout parameters from configuration
        max_rollouts_per_sample = self.config.env.rollout.n  # Max rollouts allowed per sample
        initial_rollouts_per_sample = self.config.env.rollout.initial_n  # Initial active rollouts per sample
        
        # Calculate base parameters
        num_samples = len(gen_batch.batch['input_ids'])
        max_total_rollouts = num_samples * max_rollouts_per_sample
        
        print(f"Number of Samples: {num_samples}")
        print(f"Rollout Config: Max {max_rollouts_per_sample} per sample, Initial {initial_rollouts_per_sample} per sample")
        
        # 1. Reset environments. The env count is configured based on max_total_rollouts.
        obs, infos = envs.reset()

        # 2. Pre-allocate and expand gen_batch to maximum capacity
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        if len(gen_batch.batch) != lenght_obs and self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        batch_output = None

        # 3. Initialize Masks
        # batch_active_mask determines which indices are sent to the vLLM inference engine
        batch_active_mask = np.zeros(max_total_rollouts, dtype=bool) 
        
        # Initial State: Activate only the initial rollouts for each sample
        for sample_idx in range(num_samples):
            start_idx = sample_idx * max_rollouts_per_sample
            end_idx = start_idx + initial_rollouts_per_sample
            batch_active_mask[start_idx:end_idx] = True
        
        # 4. Assign unique IDs (UIDs) for each sample grouping
        if self.config.env.rollout.n > 0: # Environment grouping enabled
            uid_batch = []
            for i in range(max_total_rollouts):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # No grouping; all rollouts share a single UID
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object) 
        
        # Initialize trajectory UIDs for active rollouts
        traj_uid = np.array([None for _ in range(max_total_rollouts)], dtype=object)
        for sample_idx in range(num_samples):
            start_idx = sample_idx * max_rollouts_per_sample
            end_idx = start_idx + initial_rollouts_per_sample
            for rollout_idx in range(start_idx, end_idx):
                traj_uid[rollout_idx] = str(uuid.uuid4())
        
        is_done = np.zeros(max_total_rollouts, dtype=bool)
        episode_lengths = np.zeros(max_total_rollouts, dtype=np.int32)
        episode_rewards = np.zeros(max_total_rollouts, dtype=np.float32)
        
        total_batch_list = [[] for _ in range(max_total_rollouts)]
        total_infos = [[] for _ in range(max_total_rollouts)]
        
        # 5. Branch Management: Mapping rollout indices to parent samples
        sample_to_rollouts_map = {}  # sample_idx -> [rollout_indices]
        rollout_to_sample_map = {}   # rollout_idx -> sample_idx
        
        for sample_idx in range(num_samples):
            sample_to_rollouts_map[sample_idx] = []
            start_idx = sample_idx * max_rollouts_per_sample
            end_idx = start_idx + initial_rollouts_per_sample
            
            for rollout_idx in range(start_idx, end_idx):
                sample_to_rollouts_map[sample_idx].append(rollout_idx)
                rollout_to_sample_map[rollout_idx] = sample_idx
        
        # Branch Metadata: Tracking parent-child relationships for the search tree
        branch_info = {
            'branch_points': [],
            'parent_child_map': {}, # Child -> Parent
            'branch_tree': {}       # Parent -> [Children List]
        }

        # Track the next available slot for new branches within each sample's pre-allocated block
        next_available_rollout_per_sample = {}
        for sample_idx in range(num_samples):
            next_available_rollout_per_sample[sample_idx] = sample_idx * max_rollouts_per_sample + initial_rollouts_per_sample
        
        # Calculate initial active mask
        current_active_mask = batch_active_mask & (~is_done)
        active_indices = np.where(current_active_mask)[0]
        
        initial_entropy_dict = {}
        total_branches = 0
        step_branch_full = -1
        all_step_branches = []

        # Main Multi-turn Loop
        for ii in range(self.config.env.max_steps):
            _step = ii + 1
            
            # Extract only active parts of the batch and observations using masks
            active_batch = extract_batch_with_mask(gen_batch, active_indices)
            active_obs = extract_obs_with_mask(obs, active_indices)
            
            # Preprocess and handle model generation
            batch = self.preprocess_batch(gen_batch=active_batch, obs=active_obs)
            
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "multi_modal_data", "raw_prompt", "tools_kwargs"]
            # (Note: Pop logic ensures clean input for model generation)
            
            batch_input = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
            batch_input.meta_info = gen_batch.meta_info

            # Configure custom logprob tracking if using entropy-based branching
            if self.config.env.use_branch and self.config.env.branch_condition == "entropy":
                batch_input.meta_info.update({"logprobs_custom": 10, "logprobs_tokens_num": 20})
            
            # Perform rollout generation via inference engine
            batch_output = actor_rollout_wg.generate_sequences(batch_input)
            
            if self.config.env.use_branch and self.config.env.branch_condition == "entropy":
                entropys = batch_output.non_tensor_batch.pop("entropys")

            # Fill UIDs for tracking
            batch.non_tensor_batch['uid'] = uid_batch[active_indices]
            batch.non_tensor_batch['traj_uid'] = traj_uid[active_indices]
            batch = batch.union(batch_output) 
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            batch.non_tensor_batch['full_output'] = deepcopy(text_actions)
            batch.non_tensor_batch['node_uid'] = [uuid.uuid4() for i in range(len(text_actions))]

            # Categorize the reasoning mode (Planning, Exploration, or standard Thinking)
            def extract_reasoning_modes(text_actions):
                res = []
                for output in text_actions:
                    if '<planning>' in output: res.append('<planning>')
                    elif '<explore>' in output: res.append('<explore>')
                    else: res.append('<think>')
                return res
            batch.non_tensor_batch['reasoning_modes'] = extract_reasoning_modes(text_actions)

            # Step the environment using the active mask
            next_obs, rewards, dones, next_infos = envs.step_with_mask(text_actions, active_indices)

            # Standardize rewards and update episode stats
            if len(rewards.shape) == 2: rewards = rewards.squeeze(1)
            if len(dones.shape) == 2: dones = dones.squeeze(1)

            # Track action validity and availability from env feedback
            batch.non_tensor_batch['is_action_valid'] = np.array([info.get('is_action_valid', True) for info in next_infos], dtype=bool)
            batch.non_tensor_batch['is_action_available'] = np.array([info.get('is_action_available', True) for info in next_infos], dtype=bool)

            episode_rewards[active_indices] += torch_to_numpy(rewards)
            episode_lengths[active_indices] += 1

            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = np.ones(len(active_indices), dtype=bool)
            
            batch_list = to_list_of_dict(batch)

            # Store results back to the respective trajectory buffers
            for i, rollout_idx in enumerate(active_indices):
                total_batch_list[rollout_idx].append(batch_list[i])
                total_infos[rollout_idx].append(next_infos[i])

            # Update done status
            is_done[active_indices] = np.logical_or(is_done[active_indices], dones)
            
            # 6. Branch Decision Logic
            if self.config.env.use_branch and self.config.env.branch_condition == "entropy":
                current_entropy_dict = entropy_variation_monitoring(entropys, active_indices, self.tokenizer, initial_entropy_dict)
                should_branch = adaptive_entropy_branching_condition(_step, initial_entropy_dict, current_entropy_dict, active_indices, branch_probability=float(self.config.env.branch_probability))
            elif self.config.env.use_branch and self.config.env.branch_condition == "explore":
                should_branch = explore_branching_condition(_step, batch.non_tensor_batch['full_output'])
            elif self.config.env.use_branch and self.config.env.branch_condition == "fixed":
                should_branch = fixed_branching_condition(_step, batch.non_tensor_batch['full_output'], num_branches=4)
            else:
                should_branch = np.zeros(len(active_indices), dtype=bool)

            # 7. Execute Branching (Cloning states to new rollout slots)
            this_step_branches = 0
            all_branches = {}
            if np.any(should_branch):
                branch_indices = active_indices[should_branch]
                random.shuffle(branch_indices)
                
                for branch_idx in branch_indices:
                    sample_idx = rollout_to_sample_map[branch_idx]
                    current_rollout_count = len(sample_to_rollouts_map[sample_idx])
                    available_slots = max_rollouts_per_sample - current_rollout_count
                    actual_branch_count = min(branch_count - 1, available_slots)
                    
                    if actual_branch_count > 0:
                        parent_traj_uid = traj_uid[branch_idx]
                        parent_uid = uid_batch[branch_idx]
                        branch_rollout_indices = []
                        
                        for i in range(actual_branch_count):
                            new_rollout_idx = next_available_rollout_per_sample[sample_idx]
                            
                            # Ensure we don't exceed allocated memory for this sample
                            sample_rollout_end = (sample_idx + 1) * max_rollouts_per_sample
                            if new_rollout_idx >= sample_rollout_end: break
                            
                            next_available_rollout_per_sample[sample_idx] += 1
                            batch_active_mask[new_rollout_idx] = True
                            
                            new_traj_uid = str(uuid.uuid4())
                            traj_uid[new_rollout_idx] = new_traj_uid
                            uid_batch[new_rollout_idx] = parent_uid
                            
                            # Deep copy history and state from parent to child
                            total_batch_list[new_rollout_idx] = deepcopy(total_batch_list[branch_idx])
                            total_infos[new_rollout_idx] = deepcopy(total_infos[branch_idx])
                            
                            # Update trajectory UID in the history of the new branch
                            for step_idx in range(len(total_batch_list[new_rollout_idx])):
                                total_batch_list[new_rollout_idx][step_idx]['traj_uid'] = new_traj_uid

                            episode_lengths[new_rollout_idx] = episode_lengths[branch_idx]
                            episode_rewards[new_rollout_idx] = episode_rewards[branch_idx]
                            is_done[new_rollout_idx] = is_done[branch_idx]
                            
                            sample_to_rollouts_map[sample_idx].append(new_rollout_idx)
                            rollout_to_sample_map[new_rollout_idx] = sample_idx
                            
                            branch_info['parent_child_map'][new_traj_uid] = parent_traj_uid
                            branch_info['branch_tree'].setdefault(parent_traj_uid, []).append(new_traj_uid)
                            branch_rollout_indices.append(new_rollout_idx)
                        
                        if branch_rollout_indices:
                            all_branches.update({branch_idx: branch_rollout_indices})
                            branch_info['branch_points'].append({
                                'step': _step,
                                'sample_idx': int(sample_idx),
                                'parent_idx': int(branch_idx),
                                'branch_indices': [int(i) for i in branch_rollout_indices]
                            })
                            total_branches += len(branch_rollout_indices)
                            this_step_branches += len(branch_rollout_indices)
                
                # Restore/Synchronize environment backend state for new branches
                envs.sync_branch_states(all_branches)
            
            print(f"Step {_step}: {this_step_branches} branches created and synchronized.")
            all_step_branches.append(this_step_branches)

            # Update observations for both active and newly branched rollouts
            obs = update_obs_with_mask(obs, next_obs, active_indices, all_branches)  
            infos = update_infos_with_mask(infos, next_infos, active_indices, all_branches)

            # Refresh active mask for next iteration
            current_active_mask = batch_active_mask & (~is_done)
            active_indices = np.where(current_active_mask)[0]
            
            # Tracking saturation of allocated slots
            if (step_branch_full == -1) and (total_branches >= max_total_rollouts - initial_rollouts_per_sample*num_samples):
                step_branch_full = _step + 1
                print(f"ROLLOUT SLOTS FULL AT STEP: {step_branch_full}")

            if len(active_indices) == 0: break
        
        # 8. Post-processing and data aggregation
        used_rollout_mask = np.array([traj_uid[i] is not None for i in range(max_total_rollouts)])
        used_indices = np.where(used_rollout_mask)[0]
        
        final_total_batch_list = [total_batch_list[i] for i in used_indices]
        final_episode_rewards = episode_rewards[used_indices]
        final_episode_lengths = episode_lengths[used_indices]
        final_traj_uid = traj_uid[used_indices]
        final_total_infos = [total_infos[i] for i in used_indices]

        # Annotate win status in batch data for subsequent reward processing
        final_wons = [reward > 0 for reward in final_episode_rewards]
        for traj_idx, traj in enumerate(final_total_batch_list):
            for step_idx, step_data in enumerate(traj):
                step_data['won'] = int(final_wons[traj_idx])
                
        # Optional: Pad trajectories to ensure fixed-size batches for training
        if self.config.env.use_pad:
            final_total_batch_list, final_episode_rewards, final_episode_lengths, final_traj_uid, final_total_infos = \
                pad_and_rebuild_all_trajectory_data(...)

        # Final evaluation and performance metric gathering
        success = envs.success_evaluator(
            total_infos=final_total_infos, 
            total_batch_list=final_total_batch_list, 
            episode_rewards=final_episode_rewards, 
            episode_lengths=final_episode_lengths
        )
        success['custom/total_branches'] = np.array([total_branches])
        success['custom/step_branch_full'] = np.array([step_branch_full])

        print(f"Collected total of {len(final_total_batch_list)} rollout trajectories.")
        
        # Store branch lineage for analysis
        self.branch_info = branch_info
        self.all_step_branches = all_step_branches
        
        return final_total_batch_list, final_episode_rewards, final_episode_lengths, success, final_traj_uid