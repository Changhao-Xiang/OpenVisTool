import json
from pathlib import Path

root = Path("/mnt/afs_reason/xiangchanghao/Sensetime/nanobot_for_distill")
input_a = root / "workspaces/qwen35plus/AgentNet_ubuntu_filtered_index.jsonl"
input_b = root / "dataset/GUI/AgentNet_processed/ubuntu_click_highres_difficulty_filtered_qwen35_9b_tool_gain_prefix.jsonl"
output = root / "workspaces/qwen35plus/AgentNet_ubuntu_filtered_index_with_difficulty_filtered.jsonl"

ids_b = set()
with input_b.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ids_b.add(json.loads(line)["id"])

count_a = 0
count_out = 0
with input_a.open("r", encoding="utf-8") as fin, output.open("w", encoding="utf-8") as fout:
    for line in fin:
        raw = line.strip()
        if not raw:
            continue
        count_a += 1
        obj = json.loads(raw)
        if obj["id"] in ids_b:
            fout.write(line if line.endswith("\n") else line + "\n")
            count_out += 1

print(f"input_a={count_a}")
print(f"ids_b={len(ids_b)}")
print(f"output={count_out}")
print(output)
