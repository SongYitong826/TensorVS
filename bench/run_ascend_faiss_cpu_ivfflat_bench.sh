#!/usr/bin/env bash
# One method/dataset; CPU functions from the installed Faiss Ascend package.
set -euo pipefail
if (( $# < 2 )) || [[ "$1" != ivfflat ]]; then
    echo "Usage: bash bench/run_ascend_faiss_cpu_ivfflat_bench.sh ivfflat {glove1.2m|hunyuan} [options]" >&2
    exit 2
fi
dataset="$2"
case "$dataset" in glove1.2m|hunyuan) ;; *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;; esac
shift 2
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="${FAISS_ASCEND_RUNTIME:-$(dirname "$project_root")/faiss-ascend-runtime}"
if [[ -d "$(dirname "$project_root")/envs/tensorvs/conda-meta" ]]; then
    export CONDA_ENVS_PATH="$(dirname "$project_root")/envs${CONDA_ENVS_PATH:+:$CONDA_ENVS_PATH}"
fi
if [[ ! -f "$runtime/env.sh" ]]; then
    echo "Missing Faiss Ascend runtime: $runtime/env.sh" >&2
    exit 1
fi
set +u
source "$runtime/env.sh"
set -u
# Faiss controls OpenMP threads via --cpu-threads; avoid nested BLAS threading.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="${FAISS_ASCEND_PYTHON:-$runtime/python}${PYTHONPATH:+:$PYTHONPATH}"
exec conda run --no-capture-output -n tensorvs python -u \
    "$project_root/bench/ascend_faiss_cpu_ivfflat_bench.py" \
    --dataset "$dataset" --require-ascend-package "$@"
