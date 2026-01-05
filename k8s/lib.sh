#!/bin/bash
# Common k8s job submission utilities

# Capture local env vars to pass through
env_exports=""
for var in HF_HOME HF_DATASETS_OFFLINE; do
    [ -n "${!var:-}" ] && env_exports+="export $var=\"${!var}\"; "
done

submit_job() {
    local job_name="$1"
    local run_cmd="$2"
    local job_name_safe="${job_name//_/--}"

    local cmd="
set -euo pipefail
$env_exports
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.local/bin/env
cd /data/tomp/on-policy-distillation/
source .env
uv sync
$run_cmd
"
    local cmd_b64=$(printf '%s' "$cmd" | base64)
    JOB_NAME=$job_name_safe krun --gpu 8 --priority low --run-command "echo $cmd_b64 | base64 -d | bash"
}
