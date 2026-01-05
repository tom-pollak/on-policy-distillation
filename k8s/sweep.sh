#!/bin/bash
set -euo pipefail
source ~/.bashrc

config="$1"
agents="${2:-1}"

# Create sweep and capture the sweep path (entity/project/sweep_id)
echo "Creating sweep from $config..."
sweep_output=$(wandb sweep "$config" 2>&1)
sweep_path=$(echo "$sweep_output" | grep -oP 'Run sweep agent with: wandb agent \K[^\s]+')

if [ -z "$sweep_path" ]; then
    echo "Failed to parse sweep path from output:"
    echo "$sweep_output"
    exit 1
fi

echo "Created sweep: $sweep_path"
sweep_id=$(basename "$sweep_path")

# Spawn N agents on K8s
for i in $(seq 1 "$agents"); do
    job_name="sweep-${sweep_id}-agent-${i}"
    job_name_safe="${job_name//_/--}"

    cmd="
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.local/bin/env
cd /data/tomp/on-policy-distillation/
source .env
uv sync
wandb agent $sweep_path
"

    cmd_b64=$(printf '%s' "$cmd" | base64)
    echo "Submitting agent $i: $job_name_safe"
    JOB_NAME=$job_name_safe krun --gpu 8 --priority low --run-command "echo $cmd_b64 | base64 -d | bash"
done

echo "Submitted $agents agents for sweep $sweep_path"
