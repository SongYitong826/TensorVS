#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"

for dataset in glove1.2m hunyuan; do
  bash bench/hipvs_python.sh -u bench/amd_ivfflat_build.py \
    --method hipvs --dataset "$dataset" "$@"
  /root/faiss-rocm-venv/bin/python -u bench/amd_ivfflat_build.py \
    --method faiss-rocm --dataset "$dataset" "$@"
done
