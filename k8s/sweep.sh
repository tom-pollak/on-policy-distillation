#!/bin/bash
set -euo pipefail
source ~/.bashrc
source "$(dirname "$0")/lib.sh"

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
    echo "Submitting agent $i: sweep-${sweep_id}-agent-${i}"
    submit_job "sweep-${sweep_id}-agent-${i}" "wandb agent $sweep_path"
done

echo "Submitted $agents agents for sweep $sweep_path"
