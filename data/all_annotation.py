import concurrent.futures
import json
import jsonlines
import logging
import numpy as np
import os
import random
import re
import time
from agent_system.environments.prompts.cs_alfworld_spark import ALFWORLD_TAGGING_TEMPLATE
from agent_system.environments.prompts.cs_sciworld_spark import SCIWORLD_TAGGING_TEMPLATE
from agent_system.environments.prompts.cs_webshop_spark import WEBSHOP_TAGGING_TEMPLATE

from openai import OpenAI  # 新版导入方式
# --- Constants & Configuration ---
NUM_TRAJS = 300
MAX_WORKERS = 60  # Number of parallel threads. Adjust based on your API rate limits.
ENV_NAME = "webshop"  # Options: alfworld, sciworld, webshop
GENERALIZATION_LEVEL = "L0"  # Options: L0, L1, L2
SAVE_DIR_BASE = f"data/{ENV_NAME}_cs_{GENERALIZATION_LEVEL}"
TEACHER_RES_SAVE_PATH = os.path.join(SAVE_DIR_BASE, f"{ENV_NAME}_annotation.jsonl") # Changed filename slightly
os.makedirs(SAVE_DIR_BASE, exist_ok=True)

if ENV_NAME == "alfworld":
    TAGGING_TEMPLATE = ALFWORLD_TAGGING_TEMPLATE
elif ENV_NAME == "sciworld":
    TAGGING_TEMPLATE = SCIWORLD_TAGGING_TEMPLATE
elif ENV_NAME == "webshop":
    TAGGING_TEMPLATE = WEBSHOP_TAGGING_TEMPLATE

base_url = "https://api.moonshot.cn/v1"
api_key = "xxxx"
MODEL = "kimi-k2-0905-preview"
client = OpenAI(api_key=api_key, base_url=base_url)

# --- Load Dataset ---
print("Loading dataset...")
with open(f'data/expert_traj/{ENV_NAME}_sft_{GENERALIZATION_LEVEL}.json','r') as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} raw entries.")

trajs = []
for traj in dataset:
    conversations = traj["conversations"][2:]
    traj_log = []
    task = traj.get("task", "Unknown Task")
    for i in range(0, len(conversations), 2):
        obs = conversations[i]["value"].strip()
        llm_action = conversations[i + 1]["value"]
        action = llm_action.split("Action:")[-1].strip()
        traj_log.append({"observation": obs, "action": action})
    
    if len(traj_log) < 30:
        trajs.append({"task": task, "traj": traj_log})

print(f"Filtered to {len(trajs)} trajectories.")
random.seed(42)
random.shuffle(trajs)
print("Trajectories shuffled.")

# --- Core Functions (Unchanged) ---

def llm(prompt, model, temperature=0.0, max_tokens=1024, retries=3):
    """Synchronous LLM call with retries."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}. Retrying... (Attempt {attempt + 1}/{retries})")
            time.sleep(2 ** attempt)
    return None

def llm_json(prompt, model, temperature=0.0, max_tokens=1024, retries=5):
    """Synchronous LLM call expecting JSON, with retries."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Error: {e}. Retrying... (Attempt {attempt + 1}/{retries})")
            time.sleep(2 ** attempt)
    return []

def merge(res, traj):
    """Validates and merges LLM response with original trajectory."""
    if (len(res) != len(traj)):
        print(f"Length mismatch: {len(res)} vs {len(traj)}")
        return False, None

    if not res:
        print("Empty response from LLM.")
        return False, None

    if "action" not in res[0] or "reason" not in res[0]:
        print("Missing 'action' or 'reason' in the response structure.")
        return False, None
    
    output = []
    for a, b in zip(res, traj):
        if a["action"] != b["action"]:
            print(f"Action mismatch: {a['action']} vs {b['action']}")
            return False, None
        if not a["reason"].startswith("<"):
            print(f"Invalid reason format: {a['reason']}")
            return False, None
        output.append({
            "obs": b["observation"],
            "reason": a["reason"],
            "action": a["action"],
        })
    return True, output

# --- Parallel Processing Function ---

def process_trajectory(traj):
    """
    Function to be executed in a worker thread.
    Processes a single trajectory, calls the LLM, and validates the result.
    """
    try:
        prompt = TAGGING_TEMPLATE.format(traj=json.dumps(traj["traj"], ensure_ascii=False))
        # llm_json is thread-safe as it creates its own request
        # The 'client' object from openai v1.0+ is also thread-safe
        res = llm_json(prompt, MODEL) 

        valid, merged_data = merge(res, traj["traj"])

        if not valid:
            print(f"Invalid response for task: {traj['task']}")
            return {"task": traj["task"], "valid": False, "error": "Merge failed"}
        
        # If valid, prepare the data to be written
        data_to_write = {
            "task": traj["task"],
            "traj": merged_data
        }
        return {"task": traj["task"], "valid": True, "data": data_to_write}

    except Exception as e:
        print(f"Error processing task {traj.get('task', 'Unknown')}: {e}")
        return {"task": traj.get('task', 'Unknown'), "valid": False, "error": str(e)}

# --- Main Parallel Execution ---
print(f"Starting parallel processing with {MAX_WORKERS} workers...")
with jsonlines.open(TEACHER_RES_SAVE_PATH, mode='a') as writer:
    cnt_valid = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a dictionary to map futures to their original trajectories
        # This isn't strictly necessary but can be useful for debugging
        futures = {executor.submit(process_trajectory, traj): traj for traj in trajs}
        
        # Process results as they are completed
        for future in concurrent.futures.as_completed(futures):
            # Check if we've reached our goal *before* processing the result
            if cnt_valid >= NUM_TRAJS:
                # If so, cancel all remaining pending futures
                for f in futures:
                    if not f.done():
                        f.cancel()
                break  # Exit the as_completed loop
            
            try:
                result = future.result()  # Get the result from the thread
                
                if result and result.get("valid"):
                    writer.write(result["data"])
                    cnt_valid += 1
                    print(f"[{cnt_valid}/{NUM_TRAJS}] VALIDATED: {result['task']}")
                else:
                    # Handle cases where processing failed or validation failed
                    print(f"SKIPPED invalid/failed task: {result.get('task', 'Unknown')}")

            except concurrent.futures.CancelledError:
                # This is expected if we cancelled futures after reaching NUM_TRAJS
                print("A future was cancelled.")
            except Exception as e:
                # Catch any other unexpected error from future.result()
                traj = futures[future]  # Get the original traj using the future
                print(f"CRITICAL ERROR for task {traj.get('task', 'Unknown')}: {e}")

print(f"\nProcessing complete.")
print(f"Total valid trajectories saved: {cnt_valid}")
print(f"Data saved to: {TEACHER_RES_SAVE_PATH}")
