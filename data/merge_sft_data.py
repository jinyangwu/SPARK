import json

full_data = []

with open('data/alfworld_cs_L0/alfworld_cold-start_annotation.json', 'r') as f1:
    data1 = json.load(f1)

with open('data/alfworld_cs_L0/interaction.json', 'r') as f2:
    data2 = json.load(f2)

full_data.extend(data1)
full_data.extend(data2)

with open('data/alfworld_cs_L0/alfworld_cold-start.json', 'w') as f:
    json.dump(full_data, f, indent=4)