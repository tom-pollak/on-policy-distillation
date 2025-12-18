#!/bin/bash
set -euo pipefail
source ~/.bashrc

job_name="$1"
job_name_safe="${job_name//_/--}"
shift

cmd="
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.local/bin/env
cd /data/tomp/on-policy-distillation/
source .env
uv sync
uv run accelerate launch eval.py $@
"

cmd_b64=$(printf '%s' "$cmd" | base64)
JOB_NAME=$job_name_safe krun --gpu 8 --priority low --run-command "echo $cmd_b64 | base64 -d | bash"
