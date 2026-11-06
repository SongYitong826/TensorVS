#!/usr/bin/env bash
# Run exactly one Tensor method per invocation.
# Usage: bash bench/run_ascend_tensor_bench.sh ivfflat glove1.2m --resume
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep the work-disk environment discoverable after a Notebook restart.
env_root="$(dirname "$project_root")/envs"
if [[ -d "$env_root/tensorvs/conda-meta" ]]; then
    export CONDA_ENVS_PATH="$env_root${CONDA_ENVS_PATH:+:$CONDA_ENVS_PATH}"
fi
if (( $# < 2 )); then
    echo "Usage: bash bench/run_ascend_tensor_bench.sh {ivfflat|ivfpq} {glove1.2m|hunyuan} [options]" >&2
    exit 2
fi
kind="$1"
dataset="$2"
shift 2
case "$kind" in ivfflat|ivfpq) ;; *) echo "Unknown method: $kind" >&2; exit 2 ;; esac
case "$dataset" in glove1.2m|hunyuan) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2 ;; esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
execution_options=(--memory-budget-mb 128)
# Validated with the existing GloVe index at batches 32/128/512 and probes
# 50/80/400. These are execution settings, never index/search parameters.
if [[ "$dataset" == glove1.2m ]]; then
    execution_options=(--memory-budget-mb 512)
    if [[ "$kind" == ivfpq ]]; then
        execution_options+=(--pq-scan grouped --pq-lookup gather --query-tile-size 16
                            --candidate-chunk-size 1024 --lut-chunk-size 16)
    fi
elif [[ "$dataset" == hunyuan && "$kind" == ivfpq ]]; then
    # M384 needs a different balance of query reuse, ADC reduction width and
    # temporary storage than GloVe. Preserve all persisted index parameters.
    execution_options=(--memory-budget-mb 256 --pq-scan grouped --pq-lookup gather
                       --query-tile-size 64 --candidate-chunk-size 1024 --lut-chunk-size 32)
fi
exec conda run --no-capture-output -n tensorvs python -u \
    "$project_root/bench/ascend_tensor_bench.py" \
    --dataset "$dataset" --kind "$kind" "${execution_options[@]}" "$@"
