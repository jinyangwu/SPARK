import json
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="")

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

# Use a dictionary as an accumulator, grouping data by sample and rollout ID
grouped = defaultdict(list)

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        step_id = item.get("step_id", "")
        if not step_id:
            continue
        
        # Only process trajectories that resulted in a "won" state (successful tasks)
        if item.get("won") != 1:
            continue

        # Parse step_id (Format: sampleID_rolloutID_stepNum)
        parts = step_id.split("_")
        if len(parts) < 3:
            continue
        
        # Use the combination of sample and rollout as the grouping key
        sample_rollout = "_".join(parts[:2])
        step_num = int(parts[2])

        data_entry = {
            "step": step_num,
            "obs": item.get("obs", ""),
            "prompt": item.get("input", ""),
            "response": item.get("output", ""),
            "step_id": item.get("step_id", "")
        }

        grouped[sample_rollout].append(data_entry)

final_output = []
for sr, steps in grouped.items():
    # Sort the steps chronologically based on the step number
    steps.sort(key=lambda x: x["step"])

    # Extract the task description from the prompt of the first step
    first_prompt = steps[0]["prompt"]
    task_line = ""
    for line in first_prompt.splitlines():
        if "Your current task is: " in line:
            task_line = line.strip().replace("Your current task is: ", "").strip()
            break
    
    final_output.append({
        "task": task_line,
        "done": "True",
        "data": steps
    })

# Write the processed data to the output JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)

print(f"✅ Conversion complete. Processed {len(final_output)} samples. Saved to {output_path}")