#!/bin/bash
set -euo pipefail
source ~/.bashrc
source "$(dirname "$0")/lib.sh"

job_name="$1"; shift
submit_job "$job_name" "uv run accelerate launch train.py --output_dir dump/$job_name $*"
