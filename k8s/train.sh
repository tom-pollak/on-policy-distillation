#!/bin/bash
set -euo pipefail

job_name="$1"
shift

cmd="
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /data/tomp/on-policy-distillation/
source .env
uv sync
uv run accelerate launch train.py --output_dir $job_name $@
"

JOB_NAME=$job_name krun --gpu 8 --priority low --run-command "$cmd"
